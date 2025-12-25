import threading
import torch
import torch.nn.functional as F
from .matcha.flow_matching import BASECFM
from .configs import CFM_PARAMS
from tqdm import tqdm


def cast_all(*args, dtype):
    return [a if (not a.dtype.is_floating_point) or a.dtype == dtype else a.to(dtype) for a in args]


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, flow_cache=torch.zeros(1, 80, 0, 2)):

        raise NotImplementedError("unused, needs updating for meanflow model")

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = flow_cache.shape[2]
        if cache_size != 0:
            z[:, :, :cache_size] = flow_cache[:, :, :, 0]
            mu[:, :, :cache_size] = flow_cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        flow_cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), flow_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, meanflow=False):
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(x, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype)

        B, T = mu.size(0), x.size(2)
        x_in    = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2 * B,  1, T], device=x.device, dtype=x.dtype)
        mu_in   = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        t_in    = torch.zeros([2 * B       ], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2 * B, 80   ], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        r_in    = torch.zeros([2 * B       ], device=x.device, dtype=x.dtype)

        for t, r in zip(t_span[:-1], t_span[1:]):
            t = t.unsqueeze(dim=0)
            r = r.unsqueeze(dim=0)

            x_in[:B] = x_in[B:] = x
            mask_in[:B] = mask_in[B:] = mask
            mu_in[:B] = mu
            t_in[:B] = t_in[B:] = t
            spks_in[:B] = spks
            cond_in[:B] = cond
            r_in[:B] = r_in[B:] = r
            dxdt = self.estimator.forward(
                x=x_in, mask=mask_in, mu=mu_in, t=t_in, spks=spks_in, cond=cond_in,
                r=r_in if meanflow else None,
            )
            dxdt, cfg_dxdt = torch.split(dxdt, [B, B], dim=0)
            dxdt = ((1.0 + self.inference_cfg_rate) * dxdt - self.inference_cfg_rate * cfg_dxdt)
            dt = r - t
            x = x + dt * dxdt



        return x.to(in_dtype)

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        b, _, t = mu.shape

        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        self.rand_noise = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, noised_mels=None, meanflow=False):

        B = mu.size(0)
        z = torch.randn_like(mu)

        if noised_mels is not None:
            prompt_len = mu.size(2) - noised_mels.size(2)
            z[..., prompt_len:] = noised_mels

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if (not meanflow) and (self.t_scheduler == 'cosine'):
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        if meanflow:
            return self.basic_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, meanflow=meanflow), None

    def basic_euler(self, x, t_span, mu, mask, spks, cond):
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(x, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype)

        print("S3 Token -> Mel Inference...")
        for t, r in tqdm(zip(t_span[..., :-1], t_span[..., 1:]), total=t_span.shape[-1] - 1):
            t, r = t[None], r[None]
            dxdt = self.estimator.forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
            dt = r - t
            x = x + dt * dxdt

        return x.to(in_dtype)