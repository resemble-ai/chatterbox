use crate::error::{Error, Result};
use crate::shape::Shape;

pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

pub fn flat_to_coords(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    let strides = compute_strides(dims);
    for i in 0..dims.len() {
        if strides[i] > 0 {
            coords[i] = idx / strides[i];
            idx %= strides[i];
        }
    }
    coords
}

pub fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
}

pub fn broadcast_shapes(s1: &[usize], s2: &[usize]) -> Result<Vec<usize>> {
    let rank = s1.len().max(s2.len());
    let mut out = vec![0; rank];
    for i in 0..rank {
        let d1 = if i < s1.len() { s1[s1.len() - 1 - i] } else { 1 };
        let d2 = if i < s2.len() { s2[s2.len() - 1 - i] } else { 1 };
        if d1 == d2 || d1 == 1 || d2 == 1 {
            out[rank - 1 - i] = d1.max(d2);
        } else {
            return Err(Error::Msg(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                s1, s2
            )));
        }
    }
    Ok(out)
}

pub fn broadcast_binary_op<F>(
    d1: &[f32],
    s1: &[usize],
    d2: &[f32],
    s2: &[usize],
    op: F,
) -> Result<(Vec<f32>, Shape)>
where
    F: Fn(f32, f32) -> f32,
{
    let out_shape = broadcast_shapes(s1, s2)?;
    let out_len: usize = out_shape.iter().product();
    let mut out_data = Vec::with_capacity(out_len);

    let strides_out = compute_strides(&out_shape);
    let strides1 = compute_strides(s1);
    let strides2 = compute_strides(s2);

    for idx in 0..out_len {
        let coords = flat_to_coords(idx, &out_shape);

        // Map coords to s1
        let mut idx1 = 0;
        for (i, &c) in coords.iter().enumerate() {
            let offset_from_right = out_shape.len() - 1 - i;
            if offset_from_right < s1.len() {
                let dim_idx = s1.len() - 1 - offset_from_right;
                let c_mapped = if s1[dim_idx] == 1 { 0 } else { c };
                idx1 += c_mapped * strides1[dim_idx];
            }
        }

        // Map coords to s2
        let mut idx2 = 0;
        for (i, &c) in coords.iter().enumerate() {
            let offset_from_right = out_shape.len() - 1 - i;
            if offset_from_right < s2.len() {
                let dim_idx = s2.len() - 1 - offset_from_right;
                let c_mapped = if s2[dim_idx] == 1 { 0 } else { c };
                idx2 += c_mapped * strides2[dim_idx];
            }
        }

        let v1 = d1[idx1];
        let v2 = d2[idx2];
        out_data.push(op(v1, v2));
    }

    let _ = strides_out;
    Ok((out_data, Shape::new(out_shape)))
}

pub fn matmul_2d(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    out
}
