//! Rust port of `src/chatterbox/models/s3gen/matcha/decoder.py`.
//!
//! Only the leaf building blocks are ported here (`SinusoidalPosEmb`, `Block1D`,
//! `ResnetBlock1D`, `Downsample1D`, `TimestepEmbedding`, `Upsample1D`). The top
//! level `Decoder` lives in `crate::s3gen::decoder` (`ConditionalDecoder`),
//! which also carries the causal/non-causal variants.
//!
//! Weight names mirror the PyTorch module tree so a converted checkpoint loads
//! unchanged, e.g. `Block1D` -> `block.0.weight`, `block.1.weight`,
//! `ResnetBlock1D` -> `mlp.1.weight`, `block1.block.0.weight`, `res_conv.weight`.
//!
//! `candle_nn` (vendored) has no `GroupNorm`, `SiLU` or `Mish`, so they are
//! implemented here from first principles.

use candle_core::{Error, Module, Result, Tensor};
use candle_nn::{
    conv1d, conv_transpose1d, linear, ops::sigmoid, Conv1d, Conv1dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, Init, Linear, VarBuilder,
};

/// `nn.Mish()`: `x * tanh(softplus(x))`.
fn mish(x: &Tensor) -> Result<Tensor> {
    let softplus = x.exp()?.affine(1.0, 1.0)?.ln()?;
    x.broadcast_mul(&softplus.tanh()?)
}

/// `nn.SiLU()`: `x * sigmoid(x)`.
fn silu(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&sigmoid(x)?)
}

/// `nn.GroupNorm(groups, num_channels)` over a `(batch, channels, time)` tensor.
///
/// Normalises over the trailing `(channels / groups, time)` slice of each group,
/// using the biased variance, then applies per-channel affine.
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: usize,
    eps: f64,
}

impl GroupNorm {
    pub fn new(num_channels: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        // PyTorch `Decoder.initialize_weights` sets GroupNorm weight to 1, bias to 0.
        let weight = vb.get_with_hints((num_channels,), "weight", Init::Const(1.0))?;
        let bias = vb.get_with_hints((num_channels,), "bias", Init::Const(0.0))?;
        Ok(Self {
            weight,
            bias,
            groups,
            eps: 1e-5,
        })
    }
}

impl Module for GroupNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        if c % self.groups != 0 {
            return Err(Error::Msg(format!(
                "GroupNorm: channels {c} not divisible by groups {}",
                self.groups
            )));
        }
        let per_group = c / self.groups;
        // (b, c, t) -> (b, groups, per_group * t) so one reduce covers a whole group.
        let grouped = x.reshape((b, self.groups, per_group * t))?;
        let centered = grouped.broadcast_sub(&grouped.mean_keepdim(2)?)?;
        let variance = centered.sqr()?.mean_keepdim(2)?;
        let normalised = centered.broadcast_div(&variance.affine(1.0, self.eps)?.sqrt()?)?;
        let out = normalised.reshape((b, c, t))?;
        out.broadcast_mul(&self.weight.reshape((1, c, 1))?)?
            .broadcast_add(&self.bias.reshape((1, c, 1))?)
    }
}

/// Nearest-neighbour 2x upsample of a `(batch, channels, time)` tensor.
fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    x.unsqueeze(3)?
        .broadcast_as((b, c, t, 2))?
        .contiguous()?
        .reshape((b, c, 2 * t))
}

/// `SinusoidalPosEmb`, the standard transformer timestep embedding.
///
/// Consumes `t` of shape `(batch,)` and returns `(batch, dim)` holding
/// `sin` in the first half and `cos` in the second.
pub struct SinusoidalPosEmb {
    dim: usize,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Self {
        // Python asserts `dim % 2 == 0`; `dim >= 4` additionally keeps
        // `half_dim - 1` in the denominator non-zero.
        assert!(
            dim.is_multiple_of(2),
            "SinusoidalPosEmb requires dim to be even"
        );
        assert!(dim >= 4, "SinusoidalPosEmb requires dim >= 4");
        Self { dim }
    }

    /// Python default `scale=1000`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.dims().is_empty() {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let half_dim = self.dim / 2;
        // emb[i] = exp(-i * ln(10000) / (half_dim - 1))
        let factor = 10_000f64.ln() / (half_dim - 1) as f64;
        let emb_data: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f64) * factor).exp() as f32)
            .collect();
        let emb = Tensor::from_slice(&emb_data, half_dim, x.device())?;
        // (batch, 1) * (half_dim,) -> (batch, half_dim)
        let angles = x.unsqueeze(1)?.affine(1000.0, 0.0)?.broadcast_mul(&emb)?;
        let last = angles.dims().len() - 1;
        Tensor::cat(&[&angles.sin()?, &angles.cos()?], last)
    }
}

/// `Block1D`: `Conv1d(dim, dim_out, 3, padding=1)` -> `GroupNorm` -> `Mish`,
/// masking the input and the output.
pub struct Block1D {
    conv: Conv1d,
    norm: GroupNorm,
}

impl Block1D {
    pub fn new(dim: usize, dim_out: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let block = vb.pp("block");
        let conv = conv1d(
            dim,
            dim_out,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            block.pp("0"),
        )?;
        let norm = GroupNorm::new(dim_out, groups, block.pp("1"))?;
        Ok(Self { conv, norm })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(&x.broadcast_mul(mask)?)?;
        let out = mish(&self.norm.forward(&out)?)?;
        out.broadcast_mul(mask)
    }
}

/// `ResnetBlock1D`: two `Block1D`s with a timestep-conditioning projection in
/// between, plus a 1x1 `res_conv` skip that is applied unconditionally.
pub struct ResnetBlock1D {
    time_proj: Linear,
    block1: Block1D,
    block2: Block1D,
    res_conv: Conv1d,
}

impl ResnetBlock1D {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        time_emb_dim: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Python: self.mlp = Sequential(Mish(), Linear(time_emb_dim, dim_out))
        let time_proj = linear(time_emb_dim, dim_out, vb.pp("mlp").pp("1"))?;
        let block1 = Block1D::new(dim_in, dim_out, groups, vb.pp("block1"))?;
        let block2 = Block1D::new(dim_out, dim_out, groups, vb.pp("block2"))?;
        let res_conv = conv1d(
            dim_in,
            dim_out,
            1,
            Conv1dConfig::default(),
            vb.pp("res_conv"),
        )?;
        Ok(Self {
            time_proj,
            block1,
            block2,
            res_conv,
        })
    }

    /// `time_emb` must be `(batch, time_emb_dim)`.
    pub fn forward(&self, x: &Tensor, mask: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        let mut h = self.block1.forward(x, mask)?;
        // h += mlp(time_emb).unsqueeze(-1)
        let cond = self.time_proj.forward(&mish(time_emb)?)?.unsqueeze(2)?;
        h = h.broadcast_add(&cond)?;
        let h = self.block2.forward(&h, mask)?;
        let skip = self.res_conv.forward(&x.broadcast_mul(mask)?)?;
        h.broadcast_add(&skip)
    }
}

/// `Downsample1D`: `Conv1d(dim, dim, 3, stride=2, padding=1)`, halving `time`.
pub struct Downsample1D {
    conv: Conv1d,
}

impl Downsample1D {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv = conv1d(
            dim,
            dim,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 2,
                dilation: 1,
                groups: 1,
            },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// `TimestepEmbedding`: `Linear` -> activation -> `Linear`.
///
/// Python also accepts `out_dim`, `post_act_fn` and `cond_proj_dim`;
/// `ConditionalDecoder` passes none of them, so they are intentionally absent.
pub struct TimestepEmbedding {
    act_fn: String,
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(
        in_channels: usize,
        time_embed_dim: usize,
        act_fn: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_1 = linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self {
            act_fn: act_fn.to_string(),
            linear_1,
            linear_2,
        })
    }

    fn activate(&self, x: &Tensor) -> Result<Tensor> {
        match self.act_fn.as_str() {
            "silu" => silu(x),
            "mish" => mish(x),
            other => Err(Error::Msg(format!(
                "TimestepEmbedding: unsupported act_fn '{other}'"
            ))),
        }
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let x = self.activate(&self.linear_1.forward(sample)?)?;
        self.linear_2.forward(&x)
    }
}

/// `Upsample1D`: `ConvTranspose1d(channels, channels, 4, stride=2, padding=1)`
/// when `use_conv_transpose`, otherwise nearest-neighbour 2x upsampling.
///
/// Python's `use_conv` (an extra `Conv1d` after interpolation) has no caller in
/// `Decoder` and is not exposed here.
pub struct Upsample1D {
    use_conv_transpose: bool,
    conv: Option<ConvTranspose1d>,
}

impl Upsample1D {
    pub fn new(dim: usize, use_conv_transpose: bool, vb: VarBuilder) -> Result<Self> {
        let conv = if use_conv_transpose {
            Some(conv_transpose1d(
                dim,
                dim,
                4,
                ConvTranspose1dConfig {
                    padding: 1,
                    output_padding: 0,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("conv"),
            )?)
        } else {
            None
        };
        Ok(Self {
            use_conv_transpose,
            conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match (&self.conv, self.use_conv_transpose) {
            (Some(conv), _) => conv.forward(x),
            (None, false) => upsample_nearest_2x(x),
            (None, true) => Err(Error::Msg(
                "Upsample1D: use_conv_transpose set but no conv built".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn vb() -> VarBuilder {
        VarBuilder::zeros(DType::F32, &Device::Cpu)
    }

    /// GroupNorm must match the reference formula computed in plain Rust.
    #[test]
    fn group_norm_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (b, c, t) = (2usize, 4usize, 3usize);
        let x = Tensor::randn(0.0, 1.0, (b, c, t), &device)?;
        let gn = GroupNorm::new(c, 2, vb())?;
        let out = gn.forward(&x)?.to_vec3::<f32>()?;

        let raw = x.to_vec3::<f32>()?;
        let eps = 1e-5_f32;
        let per_group = c / 2;
        for bi in 0..b {
            for group in 0..2 {
                // Statistics are shared across the whole group...
                let first = group * per_group;
                let vals: Vec<f32> = raw[bi][first..first + per_group]
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect();
                let n = vals.len() as f32;
                let mean = vals.iter().sum::<f32>() / n;
                let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
                let denom = (var + eps).sqrt();
                // ...but each channel is scaled by its own affine params.
                for (cj, values) in raw[bi][first..first + per_group].iter().enumerate() {
                    let ci = first + cj;
                    for (tj, v) in values.iter().enumerate() {
                        let expected = (v - mean) / denom;
                        let got = out[bi][ci][tj];
                        assert!(
                            (got - expected).abs() < 1e-4,
                            "group_norm mismatch at b{bi} c{ci} t{tj}: {got} vs {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn group_norm_rejects_indivisible_groups() {
        let gn = GroupNorm::new(6, 4, vb()).unwrap();
        let x = Tensor::randn(0.0, 1.0, (1usize, 6usize, 2usize), &Device::Cpu).unwrap();
        assert!(gn.forward(&x).is_err());
    }

    /// sin fills the first half of the last dim, cos the second half.
    #[test]
    fn sinusoidal_pos_emb_layout() -> Result<()> {
        let device = Device::Cpu;
        let dim = 8usize;
        let half_dim = dim / 2;
        let times = [0.0f64, 0.5];
        let t = Tensor::new([0.0f32, 0.5], &device)?;
        let emb = SinusoidalPosEmb::new(dim).forward(&t)?.to_vec2::<f32>()?;
        assert_eq!(emb.len(), times.len());
        assert_eq!(emb[0].len(), dim);

        // Reference: angle_i = scale * t * exp(-i * ln(10000) / (half_dim - 1))
        let factor = 10_000f64.ln() / (half_dim - 1) as f64;
        for (row, tv) in emb.iter().zip(times.iter()) {
            for i in 0..half_dim {
                let angle = 1000.0 * tv * (-(i as f64) * factor).exp();
                assert!(
                    (row[i] as f64 - angle.sin()).abs() < 1e-5,
                    "sin[{i}]: {} vs {}",
                    row[i],
                    angle.sin()
                );
                assert!(
                    (row[half_dim + i] as f64 - angle.cos()).abs() < 1e-5,
                    "cos[{i}]: {} vs {}",
                    row[half_dim + i],
                    angle.cos()
                );
            }
        }
        // t = 0 -> sin(0) = 0, cos(0) = 1.
        assert!(emb[0][..half_dim].iter().all(|v| v.abs() < 1e-6));
        assert!(emb[0][half_dim..].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn downsample_halves_time() -> Result<()> {
        let device = Device::Cpu;
        let ds = Downsample1D::new(4, vb())?;
        // Matches `mask[:, :, ::2]` length: ceil(t / 2).
        for (t_in, t_out) in [(8usize, 4usize), (9, 5), (1, 1)] {
            let x = Tensor::randn(0.0, 1.0, (2usize, 4usize, t_in), &device)?;
            assert_eq!(ds.forward(&x)?.dims(), &[2, 4, t_out]);
        }
        Ok(())
    }

    #[test]
    fn upsample_doubles_time() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0, 1.0, (2usize, 4usize, 5usize), &device)?;
        let conv_t = Upsample1D::new(4, true, vb())?;
        assert_eq!(conv_t.forward(&x)?.dims(), &[2, 4, 10]);
        let nearest = Upsample1D::new(4, false, vb())?;
        assert_eq!(nearest.forward(&x)?.dims(), &[2, 4, 10]);
        Ok(())
    }

    /// Nearest upsampling must repeat each sample twice.
    #[test]
    fn upsample_nearest_replicates() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new([1.0f32, 2.0, 3.0], &device)?.reshape((1, 1, 3))?;
        let y = upsample_nearest_2x(&x)?.to_vec3::<f32>()?;
        assert_eq!(y[0][0], vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        Ok(())
    }

    #[test]
    fn block1d_shapes_and_masking() -> Result<()> {
        let device = Device::Cpu;
        let block = Block1D::new(3, 6, 2, vb())?;
        let x = Tensor::randn(0.0, 1.0, (2usize, 3usize, 7usize), &device)?;
        let mask = Tensor::ones((2, 1, 7), DType::F32, &device)?;
        assert_eq!(block.forward(&x, &mask)?.dims(), &[2, 6, 7]);

        // A zero mask must force the whole output to zero.
        let zero = Tensor::zeros((2, 1, 7), DType::F32, &device)?;
        let out = block.forward(&x, &zero)?.to_vec3::<f32>()?;
        assert!(out.iter().flatten().flatten().all(|v| *v == 0.0));
        Ok(())
    }

    #[test]
    fn resnet_block_shapes() -> Result<()> {
        let device = Device::Cpu;
        let (c_in, c_out, t, temb) = (3usize, 6usize, 5usize, 16usize);
        let block = ResnetBlock1D::new(c_in, c_out, temb, 2, vb())?;
        let x = Tensor::randn(0.0, 1.0, (2usize, c_in, t), &device)?;
        let mask = Tensor::ones((2, 1, t), DType::F32, &device)?;
        let time_emb = Tensor::randn(0.0, 1.0, (2usize, temb), &device)?;
        assert_eq!(block.forward(&x, &mask, &time_emb)?.dims(), &[2, c_out, t]);
        Ok(())
    }

    #[test]
    fn timestep_embedding_shapes_and_act() -> Result<()> {
        let device = Device::Cpu;
        for act in ["silu", "mish"] {
            let mlp = TimestepEmbedding::new(8, 32, act, vb())?;
            let out = mlp.forward(&Tensor::randn(0.0, 1.0, (2usize, 8usize), &device)?)?;
            assert_eq!(out.dims(), &[2, 32]);
        }
        let bad = TimestepEmbedding::new(8, 32, "not_an_act", vb())?;
        assert!(bad
            .forward(&Tensor::randn(0.0, 1.0, (2, 8), &device)?)
            .is_err());
        Ok(())
    }

    /// `silu` and `mish` spot-checked against their closed forms.
    #[test]
    fn activations_match_closed_form() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new([-1.0f32, 0.0, 0.5, 2.0], &device)?;
        let inputs = [-1.0f64, 0.0, 0.5, 2.0];

        // silu(x) = x * sigmoid(x); sigmoid(0) == 0.5 makes these exact.
        for (got, v) in silu(&x)?.to_vec1::<f32>()?.iter().zip(inputs.iter()) {
            let expect = v * 1.0 / (1.0 + (-v).exp());
            assert!(
                (*got as f64 - expect).abs() < 1e-5,
                "silu {got} vs {expect}"
            );
        }

        // mish(x) = x * tanh(ln(1 + e^x)); mish(0) == 0.
        for (got, v) in mish(&x)?.to_vec1::<f32>()?.iter().zip(inputs.iter()) {
            let expect = v * (1.0 + v.exp()).ln().tanh();
            assert!(
                (*got as f64 - expect).abs() < 1e-5,
                "mish {got} vs {expect}"
            );
        }
        Ok(())
    }

    /// Checkpoint weights must land under the PyTorch module paths.
    #[test]
    fn weight_names_match_python_module_tree() -> Result<()> {
        let varmap = VarMap::new();
        // Block1D: block.0.{weight,bias}, block.1.{weight,bias}
        Block1D::new(
            2,
            4,
            2,
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )?;
        // ResnetBlock1D: mlp.1, block1.block.0, block2.block.0, res_conv
        ResnetBlock1D::new(
            2,
            4,
            8,
            2,
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )?;
        TimestepEmbedding::new(
            8,
            32,
            "silu",
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )?;
        Downsample1D::new(
            4,
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )?;
        Upsample1D::new(
            4,
            true,
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )?;

        for expected in [
            "block.0.weight",
            "block.0.bias",
            "block.1.weight",
            "block.1.bias",
            "mlp.1.weight",
            "mlp.1.bias",
            "block1.block.0.weight",
            "block2.block.1.bias",
            "res_conv.weight",
            "res_conv.bias",
            "linear_1.weight",
            "linear_2.bias",
            "conv.weight",
            "conv.bias",
        ] {
            assert!(
                varmap.get(expected).is_some(),
                "missing weight path {expected}"
            );
        }
        Ok(())
    }

    /// Weights pulled from a `VarMap` must actually be used, not silently re-inited.
    #[test]
    fn loaded_weights_are_used() -> Result<()> {
        let device = Device::Cpu;
        let (c_in, c_out, t) = (3usize, 4usize, 4usize);
        let varmap = VarMap::new();
        // Zero both convs so block1/block2 emit 0 and only the skip survives.
        varmap.set(
            "block1.block.0.weight",
            Tensor::zeros((c_out, c_in, 3), DType::F32, &device)?,
        );
        varmap.set(
            "block2.block.0.weight",
            Tensor::zeros((c_out, c_out, 3), DType::F32, &device)?,
        );
        // res_conv weight all ones, bias 1 -> skip = sum(input channels) + 1.
        varmap.set(
            "res_conv.weight",
            Tensor::ones((c_out, c_in, 1), DType::F32, &device)?,
        );
        varmap.set(
            "res_conv.bias",
            Tensor::ones((c_out,), DType::F32, &device)?,
        );

        let block = ResnetBlock1D::new(
            c_in,
            c_out,
            8,
            2,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        // Channel i is constant (i + 1) across time -> sum = 1 + 2 + 3 = 6.
        let x = Tensor::new(
            [
                1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            ],
            &device,
        )?
        .reshape((1, c_in, t))?;
        let mask = Tensor::ones((1, 1, t), DType::F32, &device)?;
        let time_emb = Tensor::zeros((1usize, 8usize), DType::F32, &device)?;

        let out = block.forward(&x, &mask, &time_emb)?.to_vec3::<f32>()?;
        for (c, channel) in out[0].iter().enumerate() {
            assert_eq!(channel.len(), t);
            for (ti, v) in channel.iter().enumerate() {
                assert!((v - 7.0).abs() < 1e-5, "skip at c{c} t{ti}: {v} vs 7.0");
            }
        }
        Ok(())
    }
}
