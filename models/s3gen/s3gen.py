# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from omegaconf import DictConfig

from models.s3gen.s3tokenizer.const import S3_SR, SPEECH_VOCAB_SIZE
from models.s3gen.s3tokenizer.s3tokenizer import S3Tokenizer
from models.s3gen.const import COSY_SR
from models.s3gen.flow import CausalMaskedDiffWithXvec
from models.s3gen.xvector import CAMPPlus
from models.s3gen.utils.mel import mel_spectrogram
from models.s3gen.f0_predictor import ConvRNNF0Predictor
from models.s3gen.hifigan import HiFTGenerator
from models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder
from models.s3gen.flow_matching import CausalConditionalCFM
from models.s3gen.decoder import ConditionalDecoder


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


class S3Token2Mel(torch.nn.Module):
    """
    CFM decoder maps S3 speech tokens to mel-spectrograms.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram # TODO: make it a torch module?
        self.speaker_encoder = CAMPPlus()  # use default args

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
        )
        cfm_params = DictConfig({
            "sigma_min": 1e-06,
            "solver": 'euler',
            "t_scheduler": 'cosine',
            "training_cfg_rate": 0.2,
            "inference_cfg_rate": 0.7,
            "reg_loss_type": 'l1',
        })
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder
        )

        self.resamplers = {}

    @property
    def device(self):
        params = self.tokenizer.parameters()
        return next(params).device

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        ref_wav: torch.Tensor,
        ref_sr: int,
    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.

        NOTE:
        - The speaker encoder accepts 16 kHz waveform.
        - S3TokenizerV2 accepts 16 kHz waveform.
        - The mel-spectrogram for the reference assumes 24 kHz input signal.

        Args
        ----
        - `speech_tokens`: S3 speech tokens [B=1, T]
        - `ref_wav`: reference waveform (`torch.Tensor` with shape=[B=1, T])
        - `ref_sr`: reference sample rate
        """
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        assert speech_tokens.shape[0] == 1, "only batch size of one allowed for now"
        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)

        if ref_sr not in self.resamplers:
            self.resamplers[ref_sr] = ta.transforms.Resample(ref_sr, COSY_SR)
        resampler = self.resamplers[ref_sr].to(ref_wav.device)
        ref_wav_24 = resampler(ref_wav)
        ref_mels_24 = self.mel_extractor(ref_wav_24)
        ref_mels_24 = ref_mels_24.transpose(1, 2).to(self.device)
        ref_mels_24_len = None

        # Resample to 16kHz
        ref_wav_16 = ta.transforms.Resample(ref_sr, S3_SR)(ref_wav)
        ref_wav_16 = ref_wav_16.to(self.device)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)

        # Tokenize 16khz reference
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            prompt_token=ref_speech_tokens,
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
            finalize=False,
        )
        return output_mels


class S3Gen(S3Token2Mel):
    """
    Concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.
    """

    def __init__(self):
        super().__init__()

        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=COSY_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )
        self.n_trim = COSY_SR // 50  # 20ms = half of a frame

    def forward(self, speech_tokens, ref_wav, ref_sr):
        output_mels = super().forward(speech_tokens, ref_wav, ref_sr)

        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, cache_source=hift_cache_source)

        # Trim the first 20ms which is "spillover" from the reference clip.
        output_wavs = output_wavs[:, self.n_trim:]
        return output_wavs


if __name__ == '__main__':
    import sys
    import logging
    logging.getLogger("numba").setLevel(logging.WARNING)
    import torchaudio as ta

    model = S3Gen()
    model.eval()

    state_dict = torch.load("s3gen.pth")
    load_msg = model.load_state_dict(state_dict, strict=False)
    print(load_msg)

    input_wav_fpath = sys.argv[1]
    ref_wav_fpath = sys.argv[2]
    out_wav_fpath = sys.argv[3]

    wav, sr = ta.load(input_wav_fpath)
    wav = ta.transforms.Resample(sr, S3_SR)(wav)
    _speech_tokens, _ = model.tokenizer(wav)

    _ref_wav, _ref_sr = ta.load(ref_wav_fpath)

    wav_outputs = model.forward(_speech_tokens, _ref_wav, _ref_sr)
    ta.save(out_wav_fpath, wav_outputs.detach().cpu(), COSY_SR)
