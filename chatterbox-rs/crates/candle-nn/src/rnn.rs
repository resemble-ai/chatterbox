use candle_core::{Module, Result, Tensor};
use crate::linear::Linear;
use crate::var_map::VarBuilder;

pub trait RNN {
    type State;
    fn step(&self, input: &Tensor, state: &Self::State) -> Result<Self::State>;
    fn seq(&self, input: &Tensor) -> Result<Vec<Self::State>>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LSTMConfig {}

#[derive(Clone, Debug)]
pub struct LSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

impl LSTMState {
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    pub fn c(&self) -> &Tensor {
        &self.c
    }
}

#[derive(Clone, Debug)]
pub struct LSTM {
    w_ih: Linear,
    w_hh: Linear,
    hidden_dim: usize,
}

impl LSTM {
    pub fn new(w_ih: Linear, w_hh: Linear, hidden_dim: usize) -> Self {
        Self {
            w_ih,
            w_hh,
            hidden_dim,
        }
    }
}

impl RNN for LSTM {
    type State = LSTMState;

    fn step(&self, x: &Tensor, state: &Self::State) -> Result<Self::State> {
        let gates = (self.w_ih.forward(x)? + self.w_hh.forward(&state.h)?)?;
        let hidden = self.hidden_dim;
        let in_gate = gates.narrow(1, 0, hidden)?.affine(1.0, 0.0)?;
        let forget_gate = gates.narrow(1, hidden, hidden)?;
        let cell_gate = gates.narrow(1, 2 * hidden, hidden)?;
        let out_gate = gates.narrow(1, 3 * hidden, hidden)?;

        // Sigmoid helper: 1 / (1 + exp(-x))
        let sig = |t: &Tensor| -> Result<Tensor> {
            let neg = (-t)?;
            let exp = neg.exp()?;
            let denom = (exp + 1.0)?;
            Tensor::ones_like(&denom)?.broadcast_div(&denom)
        };

        let i = sig(&in_gate)?;
        let f = sig(&forget_gate)?;
        let g = cell_gate.tanh()?;
        let o = sig(&out_gate)?;

        let c = (f.broadcast_mul(&state.c)? + i.broadcast_mul(&g)?)?;
        let h = o.broadcast_mul(&c.tanh()?)?;

        Ok(LSTMState { h, c })
    }

    fn seq(&self, input: &Tensor) -> Result<Vec<Self::State>> {
        let (b, seq_len, _in_dim) = input.dims3()?;
        let mut curr_state = LSTMState {
            h: Tensor::zeros((b, self.hidden_dim), candle_core::DType::F32, input.device())?,
            c: Tensor::zeros((b, self.hidden_dim), candle_core::DType::F32, input.device())?,
        };

        let mut states = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = input.narrow(1, t, 1)?.squeeze(1)?;
            curr_state = self.step(&x_t, &curr_state)?;
            states.push(curr_state.clone());
        }

        Ok(states)
    }
}

pub fn lstm(in_dim: usize, hidden_dim: usize, _config: LSTMConfig, vb: VarBuilder) -> Result<LSTM> {
    let w_ih = crate::linear::linear(in_dim, 4 * hidden_dim, vb.pp("weight_ih"))?;
    let w_hh = crate::linear::linear(hidden_dim, 4 * hidden_dim, vb.pp("weight_hh"))?;
    Ok(LSTM::new(w_ih, w_hh, hidden_dim))
}
