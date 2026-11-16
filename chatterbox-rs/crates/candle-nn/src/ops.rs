use candle_core::{Result, Tensor};

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg = (-x)?;
    let exp = neg.exp()?;
    let denom = (exp + 1.0)?;
    Tensor::ones_like(&denom)?.broadcast_div(&denom)
}

pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let max = x.max_keepdim(dim)?;
    let diff = x.broadcast_sub(&max)?;
    let exp = diff.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    exp.broadcast_div(&sum)
}

pub fn dropout(x: &Tensor, _pr: f32) -> Result<Tensor> {
    Ok(x.clone())
}
