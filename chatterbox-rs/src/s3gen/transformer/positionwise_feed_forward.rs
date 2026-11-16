use candle_core::{Result, Tensor, Module};
use candle_nn::{linear, Linear, linear_no_bias, VarBuilder};

use super::activation::Swish;

pub enum Activation {
    ReLU,
    Swish(Swish),
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::ReLU => xs.relu(),
            Activation::Swish(swish) => swish.forward(xs),
        }
    }
}

pub struct PositionwiseFeedForward {
    w_1: Linear,
    activation: Activation,
    dropout_rate: f64,
    w_2: Linear,
}

impl PositionwiseFeedForward {
    pub fn new(
        idim: usize,
        hidden_units: usize,
        dropout_rate: f64,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let w_1 = linear(idim, hidden_units, vb.pp("w_1"))?;
        let w_2 = linear(hidden_units, idim, vb.pp("w_2"))?;
        Ok(Self {
            w_1,
            activation,
            dropout_rate,
            w_2,
        })
    }
    
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut out = self.w_1.forward(xs)?;
        out = self.activation.forward(&out)?;
        
        if train && self.dropout_rate > 0.0 {
            out = candle_nn::ops::dropout(&out, self.dropout_rate as f32)?;
        }
        
        self.w_2.forward(&out)
    }
}

impl Module for PositionwiseFeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, false)
    }
}


pub struct MoEFFNLayer {
    gate: Linear,
    experts: Vec<PositionwiseFeedForward>,
    n_expert_per_token: usize,
}

impl MoEFFNLayer {
    pub fn new(
        n_expert: usize,
        n_expert_per_token: usize,
        idim: usize,
        hidden_units: usize,
        dropout_rate: f64,
        activation_fn: fn() -> Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate = linear_no_bias(idim, n_expert, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(n_expert);
        let experts_vb = vb.pp("experts");
        
        for i in 0..n_expert {
            experts.push(PositionwiseFeedForward::new(
                idim,
                hidden_units,
                dropout_rate,
                activation_fn(),
                experts_vb.pp(i.to_string()),
            )?);
        }
        
        Ok(Self {
            gate,
            experts,
            n_expert_per_token,
        })
    }
    
    pub fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let xs_flat = xs.reshape((b * l, d))?;
        
        let router = self.gate.forward(&xs_flat)?; // (B*L, n_expert)
        
        // Custom topk for Candle.
        // We will do a simple sort because n_expert is usually small.
        let router_vec = router.flatten_all()?.to_vec1::<f32>()?;
        let n_expert = self.experts.len();
        
        let mut indices_data: Vec<u32> = Vec::with_capacity(b * l * self.n_expert_per_token);
        let mut topk_logits_data: Vec<f32> = Vec::with_capacity(b * l * self.n_expert_per_token);
        
        for i in 0..(b * l) {
            let start = i * n_expert;
            let end = start + n_expert;
            let mut row: Vec<(usize, f32)> = router_vec[start..end].iter().copied().enumerate().collect();
            // Sort descending
            row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            for k in 0..self.n_expert_per_token {
                indices_data.push(row[k].0 as u32);
                topk_logits_data.push(row[k].1);
            }
        }
        
        let topk_logits = Tensor::from_vec(topk_logits_data, (b * l, self.n_expert_per_token), xs.device())?;
        
        // softmax on dim 1
        let weights = candle_nn::ops::softmax(&topk_logits, 1)?;
        let weights_vec = weights.flatten_all()?.to_vec1::<f32>()?;
        
        // For output, it's zeros of shape (B*L, D)
        let mut output_vec = vec![0.0f32; b * l * d];
        let xs_vec = xs_flat.flatten_all()?.to_vec1::<f32>()?;
        
        for expert_idx in 0..n_expert {
            let mut batch_idx = Vec::new();
            let mut ith_expert_idx = Vec::new();
            
            for i in 0..(b * l) {
                for k in 0..self.n_expert_per_token {
                    let idx = i * self.n_expert_per_token + k;
                    if indices_data[idx] == expert_idx as u32 {
                        batch_idx.push(i);
                        ith_expert_idx.push(k);
                    }
                }
            }
            
            if batch_idx.is_empty() {
                continue;
            }
            
            let mut expert_in_data = Vec::with_capacity(batch_idx.len() * d);
            for &idx in &batch_idx {
                let start = idx * d;
                let end = start + d;
                expert_in_data.extend_from_slice(&xs_vec[start..end]);
            }
            let expert_in = Tensor::from_vec(expert_in_data, (batch_idx.len(), d), xs.device())?;
            
            let expert_out = self.experts[expert_idx].forward_t(&expert_in, false)?;
            let expert_out_vec = expert_out.flatten_all()?.to_vec1::<f32>()?;
            
            for (j, &idx) in batch_idx.iter().enumerate() {
                let w = weights_vec[idx * self.n_expert_per_token + ith_expert_idx[j]];
                let out_start = idx * d;
                let expert_start = j * d;
                for k in 0..d {
                    output_vec[out_start + k] += w * expert_out_vec[expert_start + k];
                }
            }
        }
        
        let output = Tensor::from_vec(output_vec, (b, l, d), xs.device())?;
        Ok(output)
    }
}

impl Module for MoEFFNLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarMap;

    #[test]
    fn test_positionwise_feed_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let pff = PositionwiseFeedForward::new(4, 8, 0.1, Activation::ReLU, vb)?;
        let xs = Tensor::randn(0.0f32, 1.0, (2, 3, 4), &device)?;
        let out = pff.forward(&xs)?;
        assert_eq!(out.dims3()?, (2, 3, 4));
        Ok(())
    }

    #[test]
    fn test_moe_ffn_layer() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let n_expert = 4;
        let n_expert_per_token = 2;
        let idim = 8;
        let hidden_units = 16;
        let dropout_rate = 0.1;
        
        let moe = MoEFFNLayer::new(
            n_expert,
            n_expert_per_token,
            idim,
            hidden_units,
            dropout_rate,
            || Activation::ReLU,
            vb
        )?;
        
        let xs = Tensor::randn(0.0f32, 1.0, (2, 5, 8), &device)?;
        let out = moe.forward(&xs)?;
        assert_eq!(out.dims3()?, (2, 5, 8));
        Ok(())
    }
}
