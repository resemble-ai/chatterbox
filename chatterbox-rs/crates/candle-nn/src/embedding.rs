use candle_core::{Module, Result, Tensor};
use crate::var_map::{Init, VarBuilder};

#[derive(Clone, Debug)]
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let (num_embeddings, embedding_dim) = self.weight.dims2()?;
        let orig_shape = indexes.dims().to_vec();
        let flat_indexes = indexes.flatten_all()?;
        let idx_vec = flat_indexes.to_vec1::<u32>()?;
        let w_vec = self.weight.to_vec1::<f32>()?;

        let mut out_data = Vec::with_capacity(idx_vec.len() * embedding_dim);
        for &id in &idx_vec {
            let id_usize = (id as usize).min(num_embeddings - 1);
            let start = id_usize * embedding_dim;
            out_data.extend_from_slice(&w_vec[start..start + embedding_dim]);
        }

        let mut out_shape = orig_shape;
        out_shape.push(embedding_dim);
        Tensor::from_vec(out_data, out_shape, indexes.device())
    }
}

pub fn embedding(num_embeddings: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Embedding> {
    let weight = vb.get_with_hints(
        (num_embeddings, embedding_dim),
        "weight",
        Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        },
    )?;
    Ok(Embedding::new(weight))
}
