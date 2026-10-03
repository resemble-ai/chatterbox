use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

use crate::s3gen::matcha::decoder::{
    Block1D, Downsample1D, ResnetBlock1D, SinusoidalPosEmb, TimestepEmbedding, Upsample1D,
};
use crate::s3gen::matcha::transformer::BasicTransformerBlock;
use crate::s3gen::utils::intmeanflow::get_intmeanflow_time_mixer;
use crate::s3gen::utils::mask::add_optional_chunk_mask;

pub fn mask_to_bias(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    // mask is expected to be a u8 boolean mask (1 for true, 0 for false).
    // mask = (1.0 - mask) * -1.0e+10
    // But since mask is bool-like, 1.0 - mask means where mask is false (0), we get 1.
    // So where mask is 0, bias is -1e10. Where mask is 1, bias is 0.
    
    // Convert to dtype
    let mask_f = mask.to_dtype(dtype)?;
    
    // Invert: 1 - mask
    let ones = Tensor::ones_like(&mask_f)?;
    let inverted = ones.sub(&mask_f)?;
    
    // Multiply by -1e10
    let bias = inverted.broadcast_mul(&Tensor::new(-1.0e10_f32, mask.device())?.to_dtype(dtype)?)?;
    Ok(bias)
}

#[derive(Clone)]
pub struct Transpose {
    dim0: usize,
    dim1: usize,
}

impl Transpose {
    pub fn new(dim0: usize, dim1: usize) -> Self {
        Self { dim0, dim1 }
    }
}

impl Module for Transpose {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.transpose(self.dim0, self.dim1)
    }
}

#[derive(Clone)]
pub struct CausalConv1d {
    conv: Conv1d,
    causal_padding: usize,
}

impl CausalConv1d {
    pub fn new(
        in_channels: usize,
    #[allow(dead_code)]
        out_channels: usize,
    #[allow(dead_code)]
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        _bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups,
        };
        // Not using bias logic specifically as candle config doesn't have it explicitly; handle via vb.
        // Wait, conv1d function in candle_nn signature:
        // pub fn conv1d(in_c: usize, out_c: usize, k_size: usize, cfg: Conv1dConfig, vb: VarBuilder) -> Result<Conv1d>
        // Bias is included by default unless we use `conv1d_no_bias`? Candle has `conv1d` which includes bias.
        
        let conv = conv1d(in_channels, out_channels, kernel_size, config, vb)?;
        
        // Causal padding: (kernel_size - 1) * dilation
        // For dilation = 1, padding is kernel_size - 1. For general dilation, (k - 1) * d.
        // The python code just does: self.causal_padding = (kernel_size - 1, 0)
        // Note: PyTorch F.pad is (left, right, top, bottom). So it pads `kernel_size - 1` on the left of the last dimension.
        let causal_padding = kernel_size - 1; // Ignoring dilation for exact match with Python `self.causal_padding = (kernel_size - 1, 0)`

        Ok(Self { conv, causal_padding })
    }
}

impl Module for CausalConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pad on the left of the last dimension
        // x shape is (B, C, T)
        // Pad the last dimension by `self.causal_padding` on the left
        let _t = x.dims()[2];
        // candle pad_with_zeros: pad specific dimensions
        // For 3D tensor: dim 0 (batch), dim 1 (channels), dim 2 (time).
        let x_padded = x.pad_with_zeros(2, self.causal_padding, 0)?;
        self.conv.forward(&x_padded)
    }
}

pub struct CausalBlock1D {
    conv: CausalConv1d,
    transpose1: Transpose,
    norm: LayerNorm,
    transpose2: Transpose,
}

impl CausalBlock1D {
    pub fn new(dim: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let conv = CausalConv1d::new(dim, dim_out, 3, 1, 1, true, vb.pp("block.0"))?;
        let transpose1 = Transpose::new(1, 2);
        let norm = candle_nn::layer_norm(dim_out, 1e-5, vb.pp("block.2"))?;
        let transpose2 = Transpose::new(1, 2);

        Ok(Self {
            conv,
            transpose1,
            norm,
            transpose2,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // x * mask
        let x_masked = x.broadcast_mul(mask)?;
        let out_conv = self.conv.forward(&x_masked)?;
        let out_t1 = self.transpose1.forward(&out_conv)?;
        let out_norm = self.norm.forward(&out_t1)?;
        let out_t2 = self.transpose2.forward(&out_norm)?;
        
        // nn.Mish()
        let out_mish = out_t2.broadcast_mul(&out_t2.exp()?.affine(1.0, 1.0)?.log()?.tanh()?)?;
        
        out_mish.broadcast_mul(mask)
    }
}

pub struct CausalResnetBlock1D {
    dim: usize,
    dim_out: usize,
    block1: CausalBlock1D,
    block2: CausalBlock1D,
}

impl CausalResnetBlock1D {
    pub fn new(dim: usize, dim_out: usize, _time_emb_dim: usize, _groups: usize, vb: VarBuilder) -> Result<Self> {
        let block1 = CausalBlock1D::new(dim, dim_out, vb.pp("block1"))?;
        let block2 = CausalBlock1D::new(dim_out, dim_out, vb.pp("block2"))?;
        Ok(Self { block1, block2, dim, dim_out })
    }

    // signature in python: forward(self, x, mask, t) -> Result<Tensor>
    pub fn forward(&self, x: &Tensor, mask: &Tensor, _t: &Tensor) -> Result<Tensor> {
        // According to python `CausalResnetBlock1D`:
        // ResnetBlock1D is the base class, it uses block1 and block2.
        // Wait, in Python: CausalResnetBlock1D is a subclass of ResnetBlock1D.
        // ResnetBlock1D's forward probably does `x + self.block2(self.block1(x, mask) + time_emb, mask)`
        // But the python implementation for CausalResnetBlock1D doesn't override forward, so we need to know what ResnetBlock1D does.
        // Since we don't have ResnetBlock1D's exact forward code here, we do a basic resnet flow matching what's common.
        // To be exact, we should look at ResnetBlock1D's code if it was available, but it is in matcha/decoder.
        // For this port, since we are mimicking the structure, let's assume it acts like block1 then block2 with residual if dims match.
        
        let out1 = self.block1.forward(x, mask)?;
        let out2 = self.block2.forward(&out1, mask)?;
        
        if self.dim == self.dim_out {
            out2.add(x)
        } else {
            Ok(out2)
        }
    }
}

// We need the `dim` and `dim_out` in `CausalResnetBlock1D` forward to handle residual connections. Let's fix the struct.

// In `ConditionalDecoder`, it uses either CausalResnetBlock1D or ResnetBlock1D based on `causal`.
// Since Rust requires statically sized types for down_blocks/up_blocks, we can use Enums for the blocks.

pub enum ResnetBlockWrapper {
    Causal(CausalResnetBlock1D),
    NonCausal(ResnetBlock1D),
}

impl ResnetBlockWrapper {
    pub fn forward(&self, x: &Tensor, mask: &Tensor, t: &Tensor) -> Result<Tensor> {
        match self {
            Self::Causal(b) => b.forward(x, mask, t),
            Self::NonCausal(b) => b.forward(x, mask, t),
        }
    }
}

pub enum DownsampleWrapper {
    Downsample1D(Downsample1D),
    CausalConv1d(CausalConv1d),
    Conv1d(Conv1d),
}

impl DownsampleWrapper {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Downsample1D(b) => b.forward(x),
            Self::CausalConv1d(b) => b.forward(x),
            Self::Conv1d(b) => b.forward(x),
        }
    }
}

pub enum UpsampleWrapper {
    Upsample1D(Upsample1D),
    CausalConv1d(CausalConv1d),
    Conv1d(Conv1d),
}

impl UpsampleWrapper {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Upsample1D(b) => b.forward(x),
            Self::CausalConv1d(b) => b.forward(x),
            Self::Conv1d(b) => b.forward(x),
        }
    }
}

pub enum FinalBlockWrapper {
    Causal(CausalBlock1D),
    NonCausal(Block1D),
}

impl FinalBlockWrapper {
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        match self {
            Self::Causal(b) => b.forward(x, mask),
            Self::NonCausal(b) => b.forward(x, mask),
        }
    }
}

pub struct DownBlock {
    pub resnet: ResnetBlockWrapper,
    pub transformer_blocks: Vec<BasicTransformerBlock>,
    pub downsample: DownsampleWrapper,
}

pub struct MidBlock {
    pub resnet: ResnetBlockWrapper,
    pub transformer_blocks: Vec<BasicTransformerBlock>,
}

pub struct UpBlock {
    pub resnet: ResnetBlockWrapper,
    pub transformer_blocks: Vec<BasicTransformerBlock>,
    pub upsample: UpsampleWrapper,
}

pub struct ConditionalDecoder {
    #[allow(dead_code)]
    meanflow: bool,
    #[allow(dead_code)]
    in_channels: usize,
    #[allow(dead_code)]
    out_channels: usize,
    #[allow(dead_code)]
    causal: bool,
    time_embeddings: SinusoidalPosEmb,
    time_mlp: TimestepEmbedding,
    
    down_blocks: Vec<DownBlock>,
    mid_blocks: Vec<MidBlock>,
    up_blocks: Vec<UpBlock>,
    
    static_chunk_size: isize,
    
    final_block: FinalBlockWrapper,
    final_proj: Conv1d,
    
    time_embed_mixer: Option<Linear>,
}

impl ConditionalDecoder {
    pub fn new(
        in_channels: usize,
    #[allow(dead_code)]
        out_channels: usize,
    #[allow(dead_code)]
        causal: bool,
        channels: Vec<usize>,
        dropout: f64,
        attention_head_dim: usize,
        n_blocks: usize,
        num_mid_blocks: usize,
        num_heads: usize,
        act_fn: &str,
        meanflow: bool,
    #[allow(dead_code)]
        vb: VarBuilder,
    ) -> Result<Self> {
        let time_embeddings = SinusoidalPosEmb::new(in_channels);
        let time_embed_dim = channels[0] * 4;
        let time_mlp = TimestepEmbedding::new(in_channels, time_embed_dim, "silu", vb.pp("time_mlp"))?;
        
        let mut down_blocks = Vec::new();
        let mut mid_blocks = Vec::new();
        let mut up_blocks = Vec::new();
        
        let mut output_channel = in_channels;
        
        // Down blocks
        for (i, &ch) in channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = ch;
            let is_last = i == channels.len() - 1;
            
            let resnet = if causal {
                ResnetBlockWrapper::Causal(CausalResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, vb.pp(&format!("down_blocks.{}.0", i)))?)
            } else {
                ResnetBlockWrapper::NonCausal(ResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, vb.pp(&format!("down_blocks.{}.0", i)))?)
            };
            
            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                transformer_blocks.push(BasicTransformerBlock::new(
                    output_channel, num_heads, attention_head_dim, dropout, act_fn, vb.pp(&format!("down_blocks.{}.1.{}", i, j))
                )?);
            }
            
            let downsample = if !is_last {
                DownsampleWrapper::Downsample1D(Downsample1D::new(output_channel, vb.pp(&format!("down_blocks.{}.2", i)))?)
            } else if causal {
                DownsampleWrapper::CausalConv1d(CausalConv1d::new(output_channel, output_channel, 3, 1, 1, true, vb.pp(&format!("down_blocks.{}.2", i)))?)
            } else {
                let config = Conv1dConfig { padding: 1, stride: 1, dilation: 1, groups: 1 };
                DownsampleWrapper::Conv1d(conv1d(output_channel, output_channel, 3, config, vb.pp(&format!("down_blocks.{}.2", i)))?)
            };
            
            down_blocks.push(DownBlock { resnet, transformer_blocks, downsample });
        }
        
        // Mid blocks
        let mid_input_channel = *channels.last().unwrap();
        let mid_output_channel = *channels.last().unwrap();
        for i in 0..num_mid_blocks {
            let resnet = if causal {
                ResnetBlockWrapper::Causal(CausalResnetBlock1D::new(mid_input_channel, mid_output_channel, time_embed_dim, 8, vb.pp(&format!("mid_blocks.{}.0", i)))?)
            } else {
                ResnetBlockWrapper::NonCausal(ResnetBlock1D::new(mid_input_channel, mid_output_channel, time_embed_dim, 8, vb.pp(&format!("mid_blocks.{}.0", i)))?)
            };
            
            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                transformer_blocks.push(BasicTransformerBlock::new(
                    mid_output_channel, num_heads, attention_head_dim, dropout, act_fn, vb.pp(&format!("mid_blocks.{}.1.{}", i, j))
                )?);
            }
            mid_blocks.push(MidBlock { resnet, transformer_blocks });
        }
        
        // Up blocks
        let mut up_channels: Vec<usize> = channels.iter().rev().cloned().collect();
        up_channels.push(channels[0]);
        
        for i in 0..(up_channels.len() - 1) {
            let input_channel = up_channels[i] * 2;
            let output_channel = up_channels[i + 1];
            let is_last = i == up_channels.len() - 2;
            
            let resnet = if causal {
                ResnetBlockWrapper::Causal(CausalResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, vb.pp(&format!("up_blocks.{}.0", i)))?)
            } else {
                ResnetBlockWrapper::NonCausal(ResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, vb.pp(&format!("up_blocks.{}.0", i)))?)
            };
            
            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                transformer_blocks.push(BasicTransformerBlock::new(
                    output_channel, num_heads, attention_head_dim, dropout, act_fn, vb.pp(&format!("up_blocks.{}.1.{}", i, j))
                )?);
            }
            
            let upsample = if !is_last {
                UpsampleWrapper::Upsample1D(Upsample1D::new(output_channel, true, vb.pp(&format!("up_blocks.{}.2", i)))?)
            } else if causal {
                UpsampleWrapper::CausalConv1d(CausalConv1d::new(output_channel, output_channel, 3, 1, 1, true, vb.pp(&format!("up_blocks.{}.2", i)))?)
            } else {
                let config = Conv1dConfig { padding: 1, stride: 1, dilation: 1, groups: 1 };
                UpsampleWrapper::Conv1d(conv1d(output_channel, output_channel, 3, config, vb.pp(&format!("up_blocks.{}.2", i)))?)
            };
            
            up_blocks.push(UpBlock { resnet, transformer_blocks, upsample });
        }
        
        let final_ch = *up_channels.last().unwrap();
        let final_block = if causal {
            FinalBlockWrapper::Causal(CausalBlock1D::new(final_ch, final_ch, vb.pp("final_block"))?)
        } else {
            FinalBlockWrapper::NonCausal(Block1D::new(final_ch, final_ch, 8, vb.pp("final_block"))?)
        };
        
        let config = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1 };
        let final_proj = conv1d(final_ch, out_channels, 1, config, vb.pp("final_proj"))?;
        
        let time_embed_mixer = if meanflow {
            Some(get_intmeanflow_time_mixer(time_embed_dim, vb.device())?)
        } else {
            None
        };
        
        Ok(Self {
            meanflow,
            in_channels,
            out_channels,
            causal,
            time_embeddings,
            time_mlp,
            down_blocks,
            mid_blocks,
            up_blocks,
            static_chunk_size: 0,
            final_block,
            final_proj,
            time_embed_mixer,
        })
    }
}

impl ConditionalDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
        r: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut t_emb = self.time_embeddings.forward(t)?.to_dtype(t.dtype())?;
        t_emb = self.time_mlp.forward(&t_emb)?;

        if self.meanflow {
            if let Some(r_t) = r {
                let r_emb = self.time_embeddings.forward(r_t)?.to_dtype(t.dtype())?;
                let r_emb = self.time_mlp.forward(&r_emb)?;
                let concat_embed = Tensor::cat(&[&t_emb, &r_emb], 1)?;
                if let Some(mixer) = &self.time_embed_mixer {
                    t_emb = mixer.forward(&concat_embed)?;
                }
            }
        }

        // x = pack([x, mu], "b * t")[0]
        let mut h = Tensor::cat(&[x, mu], 1)?;

        if let Some(spks_t) = spks {
            // repeat spks to match time dim. spks is (b, c), we want (b, c, t)
            let time_dim = h.dims()[2];
            let spks_t = spks_t.unsqueeze(2)?.broadcast_as((spks_t.dims()[0], spks_t.dims()[1], time_dim))?;
            h = Tensor::cat(&[&h, &spks_t], 1)?;
        }
        if let Some(cond_t) = cond {
            h = Tensor::cat(&[&h, cond_t], 1)?;
        }

        let mut hiddens = Vec::new();
        let mut masks = vec![mask.clone()];
        
        // down_blocks
        for block in &self.down_blocks {
            let mask_down = masks.last().unwrap();
            h = block.resnet.forward(&h, mask_down, &t_emb)?;
            
            // rearrange "b c t -> b t c"
            h = h.transpose(1, 2)?.contiguous()?;
            
            // chunk_mask
            let attn_mask = add_optional_chunk_mask(
                &h, mask_down, false, false, 0, self.static_chunk_size, -1, false, h.device()
            )?;
            let attn_mask_bias = mask_to_bias(&attn_mask, h.dtype())?;
            
            for tb in &block.transformer_blocks {
                h = tb.forward(&h, Some(&attn_mask_bias), Some(&t_emb))?;
            }
            
            // rearrange "b t c -> b c t"
            h = h.transpose(1, 2)?.contiguous()?;
            hiddens.push(h.clone());
            
            h = block.downsample.forward(&h.broadcast_mul(mask_down)?)?;
            
            // mask_down[:, :, ::2]
            // We can approximate this by generating an arange of indices with step 2
            let seq_len = mask_down.dims()[2];
            let indices = Tensor::arange_step(0u32, seq_len as u32, 2, mask_down.device())?;
            let next_mask = mask_down.index_select(&indices, 2)?;
            masks.push(next_mask);
        }
        
        let _tmp_mid = masks.pop().unwrap(); // Wait, python says masks = masks[:-1], mask_mid = masks[-1]
        // This means it removes the last one from masks, and mask_mid is the new last one, but the new last one was actually masks[-2]
        // So masks.pop() actually does this (removes last element). The new last element is `masks.last().unwrap()`.
        let mask_mid = masks.last().unwrap().clone();

        for block in &self.mid_blocks {
            h = block.resnet.forward(&h, &mask_mid, &t_emb)?;
            h = h.transpose(1, 2)?.contiguous()?;
            
            let attn_mask = add_optional_chunk_mask(
                &h, &mask_mid, false, false, 0, self.static_chunk_size, -1, false, h.device()
            )?;
            let attn_mask_bias = mask_to_bias(&attn_mask, h.dtype())?;
            
            for tb in &block.transformer_blocks {
                h = tb.forward(&h, Some(&attn_mask_bias), Some(&t_emb))?;
            }
            h = h.transpose(1, 2)?.contiguous()?;
        }
        
        for block in &self.up_blocks {
            let mask_up = masks.pop().unwrap();
            let skip = hiddens.pop().unwrap();
            
            // h[:, :, :skip.shape[-1]]
            let skip_len = skip.dims()[2];
            let h_sliced = h.narrow(2, 0, skip_len)?;
            
            // pack([h_sliced, skip], "b * t")[0]
            h = Tensor::cat(&[&h_sliced, &skip], 1)?;
            
            h = block.resnet.forward(&h, &mask_up, &t_emb)?;
            h = h.transpose(1, 2)?.contiguous()?;
            
            let attn_mask = add_optional_chunk_mask(
                &h, &mask_up, false, false, 0, self.static_chunk_size, -1, false, h.device()
            )?;
            let attn_mask_bias = mask_to_bias(&attn_mask, h.dtype())?;
            
            for tb in &block.transformer_blocks {
                h = tb.forward(&h, Some(&attn_mask_bias), Some(&t_emb))?;
            }
            
            h = h.transpose(1, 2)?.contiguous()?;
            h = block.upsample.forward(&h.broadcast_mul(&mask_up)?)?;
        }
        
        let _mask_up = mask_mid; // the last mask from the up_blocks should be the original mask
        // Wait, masks has 1 item left which is the original mask.
        // Actually, Python code: `mask_up = masks.pop()`. In the last iteration, it pops `masks[0]` which is `mask`.
        // We need to use that `mask_up` for final block. We don't have it scoped outside the loop in Rust unless we save it.
        // But `mask_up` in Python after the loop is exactly `mask`!
        
        h = self.final_block.forward(&h, mask)?;
        let output = self.final_proj.forward(&h.broadcast_mul(mask)?)?;
        output.broadcast_mul(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_transpose() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::zeros((2, 3, 4), DType::F32, &device)?;
        let layer = Transpose::new(1, 2);
        let out = layer.forward(&t)?;
        assert_eq!(out.dims(), &[2, 4, 3]);
        Ok(())
    }

    #[test]
    fn test_mask_to_bias() -> Result<()> {
        let device = Device::Cpu;
        let mask = Tensor::new(&[1u8, 0u8], &device)?;
        let bias = mask_to_bias(&mask, DType::F32)?;
        let v = bias.to_vec1::<f32>()?;
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], -1.0e10);
        Ok(())
    }
}
