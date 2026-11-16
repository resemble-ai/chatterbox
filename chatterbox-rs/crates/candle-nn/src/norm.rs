use candle_core::{Module, Result, Tensor};
use crate::var_map::{Init, VarBuilder};

#[derive(Clone, Copy, Debug)]
pub struct LayerNormConfig {
    pub eps: f64,
    pub remove_mean: bool,
    pub affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for LayerNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    config: LayerNormConfig,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f64) -> Self {
        Self {
            weight,
            bias,
            config: LayerNormConfig {
                eps,
                remove_mean: true,
                affine: true,
            },
        }
    }

    pub fn new_with_cfg(weight: Tensor, bias: Option<Tensor>, config: LayerNormConfig) -> Self {
        Self { weight, bias, config }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;
        let mut x_norm = if self.config.remove_mean {
            let mean = x.mean_keepdim(last_dim)?;
            x.broadcast_sub(&mean)?
        } else {
            x.clone()
        };

        let variance = x_norm.sqr()?.mean_keepdim(last_dim)?;
        let denom = (variance + self.config.eps)?.sqrt()?;
        x_norm = x_norm.broadcast_div(&denom)?;

        if self.config.affine {
            x_norm = x_norm.broadcast_mul(&self.weight)?;
            if let Some(ref bias) = self.bias {
                x_norm = x_norm.broadcast_add(bias)?;
            }
        }
        Ok(x_norm)
    }
}

pub fn layer_norm<C: Into<LayerNormConfig>>(size: usize, config: C, vb: VarBuilder) -> Result<LayerNorm> {
    let cfg = config.into();
    let weight = if cfg.affine {
        vb.get_with_hints((size,), "weight", Init::Const(1.0))?
    } else {
        Tensor::ones((size,), vb.dtype(), &vb.device())?
    };
    let bias = if cfg.affine {
        Some(vb.get_with_hints((size,), "bias", Init::Const(0.0))?)
    } else {
        None
    };
    Ok(LayerNorm::new_with_cfg(weight, bias, cfg))
}

#[derive(Clone, Debug)]
pub struct BatchNorm {
    weight: Tensor,
    bias: Tensor,
    _running_mean: Tensor,
    _running_var: Tensor,
    eps: f64,
}

impl BatchNorm {
    pub fn forward_t(&self, x: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(x)
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rank = x.dims().len();
        let c = self.weight.dims()[0];
        let mut target_shape = vec![1; rank];
        if rank > 1 {
            target_shape[1] = c;
        } else {
            target_shape[0] = c;
        }
        let w = self.weight.reshape(target_shape.clone())?;
        let b = self.bias.reshape(target_shape)?;

        let mut x_norm = x.clone();
        for dim in (0..rank).rev() {
            if dim != 1 {
                let mean = x_norm.mean_keepdim(dim)?;
                x_norm = x_norm.broadcast_sub(&mean)?;
                let var = x_norm.sqr()?.mean_keepdim(dim)?;
                let denom = (var + self.eps)?.sqrt()?;
                x_norm = x_norm.broadcast_div(&denom)?;
            }
        }

        let scaled = x_norm.broadcast_mul(&w)?;
        scaled.broadcast_add(&b)
    }
}

pub fn batch_norm(num_features: usize, eps: f64, vb: VarBuilder) -> Result<BatchNorm> {
    let weight = vb.get_with_hints((num_features,), "weight", Init::Const(1.0))?;
    let bias = vb.get_with_hints((num_features,), "bias", Init::Const(0.0))?;
    let running_mean = vb.get_with_hints((num_features,), "running_mean", Init::Const(0.0))?;
    let running_var = vb.get_with_hints((num_features,), "running_var", Init::Const(1.0))?;
    Ok(BatchNorm {
        weight,
        bias,
        _running_mean: running_mean,
        _running_var: running_var,
        eps,
    })
}

impl candle_core::ModuleT for BatchNorm {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(xs)
    }
}
