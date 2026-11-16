use candle_core::{Module, ModuleT, Result, Tensor};

#[derive(Clone, Debug)]
pub struct Dropout {
    pr: f32,
}

impl Dropout {
    pub fn new(pr: f32) -> Self {
        Self { pr }
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        if !train || self.pr <= 0.0 {
            Ok(x.clone())
        } else {
            let scale = 1.0 / (1.0 - self.pr);
            x.affine(scale as f64, 0.0)
        }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

impl ModuleT for Dropout {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        self.forward_t(x, train)
    }
}

pub fn dropout(pr: f32) -> Dropout {
    Dropout::new(pr)
}
