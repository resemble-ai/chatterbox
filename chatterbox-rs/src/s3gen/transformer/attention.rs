use candle_core::{Result, Tensor};
use candle_nn::{Dropout, Linear, Module, VarBuilder};

#[derive(Debug)]
pub struct MultiHeadedAttention {
    d_k: usize,
    h: usize,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    dropout: Dropout,
}

impl MultiHeadedAttention {
    pub fn new(
        n_head: usize,
        n_feat: usize,
        dropout_rate: f32,
        key_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert!(n_feat % n_head == 0);
        let d_k = n_feat / n_head;

        let linear_q = candle_nn::linear(n_feat, n_feat, vb.pp("linear_q"))?;

        let linear_k = if key_bias {
            candle_nn::linear(n_feat, n_feat, vb.pp("linear_k"))?
        } else {
            candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_k"))?
        };

        let linear_v = candle_nn::linear(n_feat, n_feat, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear(n_feat, n_feat, vb.pp("linear_out"))?;

        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            d_k,
            h: n_head,
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            dropout,
        })
    }

    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let n_batch = query.dim(0)?;
        let q = self.linear_q.forward(query)?;
        let k = self.linear_k.forward(key)?;
        let v = self.linear_v.forward(value)?;

        // view to (batch, time, head, d_k)
        let q = q.reshape((n_batch, (), self.h, self.d_k))?;
        let k = k.reshape((n_batch, (), self.h, self.d_k))?;
        let v = v.reshape((n_batch, (), self.h, self.d_k))?;

        // transpose to (batch, head, time, d_k)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        Ok((q, k, v))
    }

    pub fn forward_attention(
        &self,
        value: &Tensor,
        scores: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let n_batch = value.dim(0)?;
        let mut scores = scores.clone();

        if let Some(m) = mask {
            if m.dim(2)? > 0 {
                // mask is expected to be (batch, 1, time2) or (batch, time1, time2)
                // in PyTorch: mask = mask.unsqueeze(1).eq(0)
                let m = m.unsqueeze(1)?;
                // if mask values are 0, it means it's masked (so we fill with -inf)
                // Assuming mask is already a binary mask where 0 indicates mask

                // Truncate mask along time2 if necessary
                let time2 = scores.dim(3)?;
                let m = if m.dim(3)? > time2 {
                    m.narrow(3, 0, time2)?
                } else {
                    m
                };

                let zeros = Tensor::zeros_like(&m)?;
                let m_eq_0 = m.eq(&zeros)?; // Handle float mask equal to 0

                // Using NEG_INFINITY can cause NaNs in softmax if all are masked, so use a large negative number
                let min_value = f32::NEG_INFINITY;

                // Since candle does not have masked_fill directly, we create a filled tensor
                let min_tensor =
                    Tensor::new(min_value, scores.device())?.broadcast_as(scores.shape())?;
                scores = m_eq_0.where_cond(&min_tensor, &scores)?;
            }
        }

        let mut attn = candle_nn::ops::softmax(&scores, 3)?;

        if let Some(m) = mask {
            if m.dim(2)? > 0 {
                let m = m.unsqueeze(1)?;
                let time2 = attn.dim(3)?;
                let m = if m.dim(3)? > time2 {
                    m.narrow(3, 0, time2)?
                } else {
                    m
                };
                let zeros_m = Tensor::zeros_like(&m)?;
                let m_eq_0 = m.eq(&zeros_m)?;
                let zeros = Tensor::zeros_like(&attn)?;
                attn = m_eq_0.where_cond(&zeros, &attn)?;
            }
        }

        // p_attn = self.dropout(attn) - Dropout is only applied during training which is false by default in inference
        let p_attn = self.dropout.forward_t(&attn, false)?;

        // x = torch.matmul(p_attn, value)
        let x = p_attn.matmul(value)?; // (batch, head, time1, d_k)

        // x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        let x = x.transpose(1, 2)?.contiguous()?;
        let time1 = x.dim(1)?;
        let x = x.reshape((n_batch, time1, self.h * self.d_k))?;

        self.linear_out.forward(&x)
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (q, mut k, mut v) = self.forward_qkv(query, key, value)?;

        if let Some(c) = cache {
            if c.dim(0)? > 0 {
                let cache_size = c.dim(3)? / 2;
                let key_cache = c.narrow(3, 0, cache_size)?;
                let value_cache = c.narrow(3, cache_size, cache_size)?;

                k = Tensor::cat(&[&key_cache, &k], 2)?;
                v = Tensor::cat(&[&value_cache, &v], 2)?;
            }
        }

        let new_cache = Tensor::cat(&[&k, &v], 3)?;

        let d_k_f32 = self.d_k as f64;
        let scale = 1.0 / d_k_f32.sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * scale)?;

        let out = self.forward_attention(&v, &scores, mask)?;

        Ok((out, new_cache))
    }
}

#[derive(Debug)]
pub struct RelPositionMultiHeadedAttention {
    base: MultiHeadedAttention,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPositionMultiHeadedAttention {
    pub fn new(
        n_head: usize,
        n_feat: usize,
        dropout_rate: f32,
        key_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let base = MultiHeadedAttention::new(n_head, n_feat, dropout_rate, key_bias, vb.clone())?;

        let linear_pos = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?;

        let d_k = n_feat / n_head;

        let pos_bias_u = vb.get_with_hints(
            (n_head, d_k),
            "pos_bias_u",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            }, // In pytorch: xavier_uniform
        )?;

        let pos_bias_v = vb.get_with_hints(
            (n_head, d_k),
            "pos_bias_v",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;

        Ok(Self {
            base,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
        })
    }

    pub fn rel_shift(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: (batch, head, time1, 2*time1 - 1)
        let b = x.dim(0)?;
        let h = x.dim(1)?;
        let t1 = x.dim(2)?;
        let t2 = x.dim(3)?;

        let zero_pad = Tensor::zeros((b, h, t1, 1), x.dtype(), x.device())?;
        let x_padded = Tensor::cat(&[&zero_pad, x], 3)?;

        let x_padded = x_padded.reshape((b, h, t2 + 1, t1))?;
        let x_sliced = x_padded.narrow(2, 1, x_padded.dim(2)? - 1)?;

        // view_as(x) and narrow to time2/2 + 1
        let x_reshaped = x_sliced.reshape((b, h, t1, t2))?;

        x_reshaped.narrow(3, 0, t2 / 2 + 1)
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        pos_emb: &Tensor,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (q, mut k, mut v) = self.base.forward_qkv(query, key, value)?;
        // q in pytorch script: q.transpose(1, 2) which changes (batch, head, time1, d_k) to (batch, time1, head, d_k)
        let q = q.transpose(1, 2)?.contiguous()?;

        if let Some(c) = cache {
            if c.dim(0)? > 0 {
                let cache_size = c.dim(3)? / 2;
                let key_cache = c.narrow(3, 0, cache_size)?;
                let value_cache = c.narrow(3, cache_size, cache_size)?;

                k = Tensor::cat(&[&key_cache, &k], 2)?;
                v = Tensor::cat(&[&value_cache, &v], 2)?;
            }
        }

        let new_cache = Tensor::cat(&[&k, &v], 3)?;

        let n_batch_pos = pos_emb.dim(0)?;
        let p = self.linear_pos.forward(pos_emb)?;
        let p = p.reshape((n_batch_pos, (), self.base.h, self.base.d_k))?;
        let p = p.transpose(1, 2)?.contiguous()?; // (batch, head, time2, d_k)

        let pos_bias_u = self.pos_bias_u.unsqueeze(0)?.unsqueeze(1)?;
        let pos_bias_v = self.pos_bias_v.unsqueeze(0)?.unsqueeze(1)?;

        let q_with_bias_u = q
            .broadcast_add(&pos_bias_u)?
            .transpose(1, 2)?
            .contiguous()?; // (batch, head, time1, d_k)
        let q_with_bias_v = q
            .broadcast_add(&pos_bias_v)?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let matrix_ac = q_with_bias_u.matmul(&k_t)?;

        let p_t = p.transpose(2, 3)?.contiguous()?;
        let mut matrix_bd = q_with_bias_v.matmul(&p_t)?;

        if matrix_ac.shape() != matrix_bd.shape() {
            matrix_bd = self.rel_shift(&matrix_bd)?;
        }

        let d_k_f32 = self.base.d_k as f64;
        let scale = 1.0 / d_k_f32.sqrt();

        let scores = ((matrix_ac + matrix_bd)? * scale)?;

        let out = self.base.forward_attention(&v, &scores, mask)?;

        Ok((out, new_cache))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_multi_headed_attention() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let batch = 2;
        let time1 = 3;
        let time2 = 3;
        let n_head = 4;
        let d_k = 8;
        let n_feat = n_head * d_k;

        let mha = MultiHeadedAttention::new(n_head, n_feat, 0.0, true, vb)?;

        let query = Tensor::randn(0f32, 1f32, (batch, time1, n_feat), &device)?;
        let key = Tensor::randn(0f32, 1f32, (batch, time2, n_feat), &device)?;
        let value = Tensor::randn(0f32, 1f32, (batch, time2, n_feat), &device)?;

        let (out, new_cache) = mha.forward(&query, &key, &value, None, None)?;

        assert_eq!(out.shape().dims(), &[batch, time1, n_feat]);
        assert_eq!(new_cache.shape().dims(), &[batch, n_head, time2, d_k * 2]);

        Ok(())
    }

    #[test]
    fn test_rel_position_multi_headed_attention() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let batch = 2;
        let time1 = 3;
        let time2 = 3;
        let n_head = 4;
        let d_k = 8;
        let n_feat = n_head * d_k;

        let rmha = RelPositionMultiHeadedAttention::new(n_head, n_feat, 0.0, true, vb)?;

        let query = Tensor::randn(0f32, 1f32, (batch, time1, n_feat), &device)?;
        let key = Tensor::randn(0f32, 1f32, (batch, time2, n_feat), &device)?;
        let value = Tensor::randn(0f32, 1f32, (batch, time2, n_feat), &device)?;
        let pos_emb = Tensor::randn(0f32, 1f32, (batch, 2 * time1 - 1, n_feat), &device)?;

        let (out, new_cache) = rmha.forward(&query, &key, &value, None, &pos_emb, None)?;

        assert_eq!(out.shape().dims(), &[batch, time1, n_feat]);
        assert_eq!(new_cache.shape().dims(), &[batch, n_head, time2, d_k * 2]);

        Ok(())
    }
}
