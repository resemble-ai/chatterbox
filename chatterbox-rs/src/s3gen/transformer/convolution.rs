use candle_core::{DType, Result, Tensor};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, Module, batch_norm, BatchNorm, layer_norm, LayerNorm, LayerNormConfig};

pub enum NormType {
    BatchNorm,
    LayerNorm,
}

pub enum Norm {
    BatchNorm(BatchNorm),
    LayerNorm(LayerNorm),
}

pub struct ConvolutionModule {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: Norm,
    pointwise_conv2: Conv1d,
    lorder: usize,
    use_layer_norm: bool,
}

impl ConvolutionModule {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        norm_type: NormType,
        causal: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let (padding, lorder) = if causal {
            (0, kernel_size - 1)
        } else {
            if (kernel_size - 1) % 2 != 0 {
                candle_core::bail!("kernel_size should be an odd number for non-causal convolution");
            }
            ((kernel_size - 1) / 2, 0)
        };

        let pointwise_conv1_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let pointwise_conv1 = conv1d(channels, 2 * channels, 1, pointwise_conv1_cfg, vb.pp("pointwise_conv1"))?;

        let depthwise_conv_cfg = Conv1dConfig {
            padding,
            stride: 1,
            dilation: 1,
            groups: channels,
        };
        let depthwise_conv = conv1d(channels, channels, kernel_size, depthwise_conv_cfg, vb.pp("depthwise_conv"))?;

        let (norm, use_layer_norm) = match norm_type {
            NormType::BatchNorm => (Norm::BatchNorm(batch_norm(channels, 1e-5, vb.pp("norm"))?), false),
            NormType::LayerNorm => (Norm::LayerNorm(layer_norm(channels, LayerNormConfig::default(), vb.pp("norm"))?), true),
        };

        let pointwise_conv2_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let pointwise_conv2 = conv1d(channels, channels, 1, pointwise_conv2_cfg, vb.pp("pointwise_conv2"))?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            norm,
            pointwise_conv2,
            lorder,
            use_layer_norm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask_pad: Option<&Tensor>,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // x shape: (batch, time, channels)
        let mut x = x.transpose(1, 2)?.contiguous()?; // shape: (batch, channels, time)
        
        // mask batch padding
        if let Some(mask) = mask_pad {
            if mask.dim(2)? > 0 {
                let mask = mask.to_dtype(DType::U8)?; // Assuming boolean mask
                let inv_mask = mask.ones_like()?.sub(&mask)?;
                x = x.broadcast_mul(&inv_mask.to_dtype(x.dtype())?)?;
            }
        }

        let new_cache = if self.lorder > 0 {
            if let Some(c) = cache {
                if c.dim(2)? == 0 {
                    x = x.pad_with_zeros(2, self.lorder, 0)?
                } else {
                    x = Tensor::cat(&[c, &x], 2)?;
                }
            } else {
                x = x.pad_with_zeros(2, self.lorder, 0)?
            }
            let time = x.dim(2)?;
            x.narrow(2, time - self.lorder, self.lorder)?
        } else {
            Tensor::zeros((0, 0, 0), x.dtype(), x.device())?
        };

        // GLU mechanism
        x = self.pointwise_conv1.forward(&x)?; // (batch, 2*channel, time)
        // Split and GLU
        let chunks = x.chunk(2, 1)?;
        let a = &chunks[0];
        let b = &chunks[1];
        x = a.mul(&candle_nn::ops::sigmoid(b)?)?; // (batch, channel, time)

        // 1D Depthwise Conv
        x = self.depthwise_conv.forward(&x)?;
        
        // Norm
        if self.use_layer_norm {
            x = x.transpose(1, 2)?.contiguous()?;
        }
        x = match &self.norm {
            Norm::BatchNorm(bn) => bn.forward_t(&x, false)?,
            Norm::LayerNorm(ln) => ln.forward(&x)?,
        };
        // Swish / SiLU activation? The python code takes `activation`, defaults to ReLU.
        // Wait, Python code: `self.activation = activation`, defaults to `nn.ReLU()`.
        // Let's just use relu for now, or make it customizable.
        x = x.relu()?;
        
        if self.use_layer_norm {
            x = x.transpose(1, 2)?.contiguous()?;
        }

        x = self.pointwise_conv2.forward(&x)?;

        if let Some(mask) = mask_pad {
            if mask.dim(2)? > 0 {
                let mask = mask.to_dtype(DType::U8)?; // Assuming boolean mask
                let inv_mask = mask.ones_like()?.sub(&mask)?;
                x = x.broadcast_mul(&inv_mask.to_dtype(x.dtype())?)?;
            }
        }

        Ok((x.transpose(1, 2)?.contiguous()?, new_cache))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    

    #[test]
    fn test_convolution_module() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let channels = 16;
        let kernel_size = 3;
        
        let conv_module = ConvolutionModule::new(
            channels,
            kernel_size,
            NormType::LayerNorm,
            false,
            vb,
        )?;

        let x = Tensor::randn(0.0f32, 1.0f32, (2, 10, channels), &device)?;
        let mask = Tensor::ones((2, 1, 10), DType::U8, &device)?;

        let (out, cache) = conv_module.forward(&x, Some(&mask), None)?;

        assert_eq!(out.dims(), &[2, 10, channels]);
        assert_eq!(cache.dims(), &[0, 0, 0]);

        Ok(())
    }
}
