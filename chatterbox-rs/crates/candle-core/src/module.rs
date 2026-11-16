use crate::error::Result;
use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}
