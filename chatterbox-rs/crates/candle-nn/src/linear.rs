use candle_core::{Module, Result, Tensor};
use crate::var_map::{Init, VarBuilder};

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w_t = self.weight.transpose(0, 1)?;
        let mut out = x.matmul(&w_t)?;
        if let Some(ref bias) = self.bias {
            out = out.broadcast_add(bias)?;
        }
        Ok(out)
    }
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let weight = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        Init::Randn {
            mean: 0.0,
            stdev: bound,
        },
    )?;
    let bias = vb.get_with_hints((out_dim,), "bias", Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let weight = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        Init::Randn {
            mean: 0.0,
            stdev: bound,
        },
    )?;
    Ok(Linear::new(weight, None))
}
