use std::ops::{Add, Div, Mul, Neg, Sub};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{broadcast_binary_op, compute_strides, coords_to_flat, flat_to_coords, matmul_2d};
use crate::shape::{IntoShape, IntoShapeWithInferred, Shape};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: Vec<f32>,
    device: Device,
    dtype: DType,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Tensor {
    pub fn new<D: IntoTensorData>(data: D, device: &Device) -> Result<Self> {
        let (vec_data, shape, dtype) = data.into_tensor_data();
        let final_shape = shape.into_shape();
        if final_shape.elem_count() != vec_data.len() {
            return Err(Error::Msg(format!(
                "Shape {:?} elem count {} does not match data len {}",
                final_shape,
                final_shape.elem_count(),
                vec_data.len()
            )));
        }
        Ok(Self {
            shape: final_shape,
            data: vec_data,
            device: *device,
            dtype,
        })
    }

    pub fn from_vec<T: IntoF32 + Clone, S: IntoShape>(data: Vec<T>, shape: S, device: &Device) -> Result<Self> {
        let f32_data: Vec<f32> = data.into_iter().map(|x| x.into_f32()).collect();
        let s = shape.into_shape();
        if s.elem_count() != f32_data.len() {
            return Err(Error::Msg("Data len does not match shape".to_string()));
        }
        Ok(Self {
            shape: s,
            data: f32_data,
            device: *device,
            dtype: DType::F32,
        })
    }

    pub fn from_slice<T: IntoF32 + Clone, S: IntoShape>(data: &[T], shape: S, device: &Device) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape, device)
    }

    pub fn zeros<S: IntoShape>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let s = shape.into_shape();
        let count = s.elem_count();
        Ok(Self {
            shape: s,
            data: vec![0.0f32; count],
            device: *device,
            dtype,
        })
    }

    pub fn ones<S: IntoShape>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let s = shape.into_shape();
        let count = s.elem_count();
        Ok(Self {
            shape: s,
            data: vec![1.0f32; count],
            device: *device,
            dtype,
        })
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape.clone(), self.dtype, &self.device)
    }

    pub fn eye(dim: usize, dtype: DType, device: &Device) -> Result<Self> {
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Ok(Self {
            shape: Shape::new(vec![dim, dim]),
            data,
            device: *device,
            dtype,
        })
    }

    pub fn arange<T1: IntoF32, T2: IntoF32>(start: T1, end: T2, device: &Device) -> Result<Self> {
        Self::arange_step(start, end, 1.0f32, device)
    }

    pub fn arange_step<T1: IntoF32, T2: IntoF32, T3: IntoF32>(
        start: T1,
        end: T2,
        step: T3,
        device: &Device,
    ) -> Result<Self> {
        let s = start.into_f32();
        let e = end.into_f32();
        let st = step.into_f32();
        let mut data = Vec::new();
        let mut curr = s;
        if st > 0.0 {
            while curr < e {
                data.push(curr);
                curr += st;
            }
        } else if st < 0.0 {
            while curr > e {
                data.push(curr);
                curr += st;
            }
        }
        let len = data.len();
        Ok(Self {
            shape: Shape::new(vec![len]),
            data,
            device: *device,
            dtype: DType::F32,
        })
    }

    pub fn rand<S: IntoShape>(min: f32, max: f32, shape: S, device: &Device) -> Result<Self> {
        let s = shape.into_shape();
        let count = s.elem_count();
        let mut data = Vec::with_capacity(count);
        let mut seed: u64 = 123456789;
        let diff = max - min;
        for _ in 0..count {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((seed >> 32) as u32 as f32) / (u32::MAX as f32);
            data.push(min + u * diff);
        }
        Ok(Self {
            shape: s,
            data,
            device: *device,
            dtype: DType::F32,
        })
    }

    pub fn randn<S: IntoShape>(mean: f32, std: f32, shape: S, device: &Device) -> Result<Self> {
        let s = shape.into_shape();
        let count = s.elem_count();
        let mut data = Vec::with_capacity(count);
        let mut seed: u64 = 987654321;
        for _ in 0..(count + 1) / 2 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (((seed >> 32) as u32 as f32) + 1.0) / ((u32::MAX as f32) + 2.0);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (((seed >> 32) as u32 as f32) + 1.0) / ((u32::MAX as f32) + 2.0);

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z0 = r * theta.cos();
            let z1 = r * theta.sin();

            if data.len() < count {
                data.push(mean + std * z0);
            }
            if data.len() < count {
                data.push(mean + std * z1);
            }
        }
        Ok(Self {
            shape: s,
            data,
            device: *device,
            dtype: DType::F32,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn dim(&self, d: usize) -> Result<usize> {
        let dims = self.dims();
        if d < dims.len() {
            Ok(dims[d])
        } else {
            Err(Error::Msg(format!("Dim index {} out of range for rank {}", d, dims.len())))
        }
    }

    pub fn dims1(&self) -> Result<usize> {
        let dims = self.dims();
        if dims.len() == 1 {
            Ok(dims[0])
        } else {
            Err(Error::Msg(format!("Expected 1D tensor, got {:?}", dims)))
        }
    }

    pub fn dims2(&self) -> Result<(usize, usize)> {
        let dims = self.dims();
        if dims.len() == 2 {
            Ok((dims[0], dims[1]))
        } else {
            Err(Error::Msg(format!("Expected 2D tensor, got {:?}", dims)))
        }
    }

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        let dims = self.dims();
        if dims.len() == 3 {
            Ok((dims[0], dims[1], dims[2]))
        } else {
            Err(Error::Msg(format!("Expected 3D tensor, got {:?}", dims)))
        }
    }

    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let dims = self.dims();
        if dims.len() == 4 {
            Ok((dims[0], dims[1], dims[2], dims[3]))
        } else {
            Err(Error::Msg(format!("Expected 4D tensor, got {:?}", dims)))
        }
    }

    pub fn elem_count(&self) -> usize {
        self.data.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            shape: self.shape.clone(),
            data: self.data.clone(),
            device: self.device,
            dtype,
        })
    }

    pub fn contiguous(&self) -> Result<Self> {
        Ok(self.clone())
    }

    pub fn reshape<S: IntoShapeWithInferred>(&self, shape: S) -> Result<Self> {
        let s = shape.into_shape_with_inferred(self.elem_count())?;
        if s.elem_count() != self.elem_count() {
            return Err(Error::Msg(format!(
                "Cannot reshape from {:?} ({} elems) to {:?} ({} elems)",
                self.shape,
                self.elem_count(),
                s,
                s.elem_count()
            )));
        }
        Ok(Self {
            shape: s,
            data: self.data.clone(),
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        if dim > dims.len() {
            return Err(Error::Msg(format!("Unsqueeze dim {} out of range", dim)));
        }
        dims.insert(dim, 1);
        Ok(Self {
            shape: Shape::new(dims),
            data: self.data.clone(),
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            return Err(Error::Msg(format!("Squeeze dim {} out of range", dim)));
        }
        if dims[dim] != 1 {
            return Ok(self.clone());
        }
        let mut new_dims = dims.to_vec();
        new_dims.remove(dim);
        Ok(Self {
            shape: Shape::new(new_dims),
            data: self.data.clone(),
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn flatten_all(&self) -> Result<Self> {
        Ok(Self {
            shape: Shape::new(vec![self.data.len()]),
            data: self.data.clone(),
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let dims = self.dims();
        let rank = dims.len();
        if dim1 >= rank || dim2 >= rank {
            return Err(Error::Msg("Transpose dims out of range".to_string()));
        }
        if dim1 == dim2 {
            return Ok(self.clone());
        }

        let mut out_dims = dims.to_vec();
        out_dims.swap(dim1, dim2);
        let out_strides = compute_strides(&out_dims);

        let mut out_data = vec![0.0f32; self.data.len()];
        for (i, val) in self.data.iter().enumerate() {
            let mut coords = flat_to_coords(i, dims);
            coords.swap(dim1, dim2);
            let out_idx = coords_to_flat(&coords, &out_strides);
            out_data[out_idx] = *val;
        }

        Ok(Self {
            shape: Shape::new(out_dims),
            data: out_data,
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("Narrow dim out of bounds".to_string()));
        }
        if start + len > dims[dim] {
            return Err(Error::Msg(format!(
                "Narrow slice [{}, {}] exceeds dim size {}",
                start,
                start + len,
                dims[dim]
            )));
        }

        let mut out_dims = dims.to_vec();
        out_dims[dim] = len;
        let mut out_data = Vec::with_capacity(out_dims.iter().product());

        for (i, val) in self.data.iter().enumerate() {
            let coords = flat_to_coords(i, dims);
            if coords[dim] >= start && coords[dim] < start + len {
                out_data.push(*val);
            }
        }

        Ok(Self {
            shape: Shape::new(out_dims),
            data: out_data,
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        let dim_size = self.dim(dim)?;
        let chunk_size = (dim_size + chunks - 1) / chunks;
        let mut out = Vec::with_capacity(chunks);
        let mut start = 0;
        for _ in 0..chunks {
            if start >= dim_size {
                break;
            }
            let len = chunk_size.min(dim_size - start);
            out.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(out)
    }

    pub fn broadcast_as<S: IntoShape>(&self, shape: S) -> Result<Self> {
        let target_s = shape.into_shape();
        let ones = Self::ones(target_s, self.dtype, &self.device)?;
        self.broadcast_mul(&ones)
    }

    pub fn broadcast_add(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| a + b)?;
        Ok(Self { shape, data, device: self.device, dtype: self.dtype })
    }

    pub fn broadcast_sub(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| a - b)?;
        Ok(Self { shape, data, device: self.device, dtype: self.dtype })
    }

    pub fn broadcast_mul(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| a * b)?;
        Ok(Self { shape, data, device: self.device, dtype: self.dtype })
    }

    pub fn broadcast_div(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| a / b)?;
        Ok(Self { shape, data, device: self.device, dtype: self.dtype })
    }

    pub fn add(&self, other: &Tensor) -> Result<Self> { self.broadcast_add(other) }
    pub fn sub(&self, other: &Tensor) -> Result<Self> { self.broadcast_sub(other) }
    pub fn mul(&self, other: &Tensor) -> Result<Self> { self.broadcast_mul(other) }
    pub fn div(&self, other: &Tensor) -> Result<Self> { self.broadcast_div(other) }

    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        let a_dims = self.dims();
        let b_dims = other.dims();
        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(Error::Msg("Matmul requires at least 2D tensors".to_string()));
        }

        let m = a_dims[a_dims.len() - 2];
        let k1 = a_dims[a_dims.len() - 1];
        let k2 = b_dims[b_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];

        if k1 != k2 {
            return Err(Error::Msg(format!("Matmul inner dimension mismatch: {} != {}", k1, k2)));
        }

        let a_batch: usize = a_dims[..a_dims.len() - 2].iter().product();
        let b_batch: usize = b_dims[..b_dims.len() - 2].iter().product();

        if a_batch == 1 && b_batch == 1 {
            let out_data = matmul_2d(&self.data, m, k1, &other.data, n);
            let mut out_dims = a_dims[..a_dims.len() - 2].to_vec();
            out_dims.push(m);
            out_dims.push(n);
            return Ok(Self {
                shape: Shape::new(out_dims),
                data: out_data,
                device: self.device,
                dtype: self.dtype,
            });
        }

        let batch_count = a_batch.max(b_batch);
        let mut out_data = Vec::with_capacity(batch_count * m * n);
        for b in 0..batch_count {
            let a_slice_idx = (b % a_batch) * m * k1;
            let b_slice_idx = (b % b_batch) * k1 * n;
            let a_slice = &self.data[a_slice_idx..a_slice_idx + m * k1];
            let b_slice = &other.data[b_slice_idx..b_slice_idx + k1 * n];
            let res = matmul_2d(a_slice, m, k1, b_slice, n);
            out_data.extend_from_slice(&res);
        }

        let mut out_dims = if a_dims.len() >= b_dims.len() {
            a_dims[..a_dims.len() - 2].to_vec()
        } else {
            b_dims[..b_dims.len() - 2].to_vec()
        };
        out_dims.push(m);
        out_dims.push(n);

        Ok(Self {
            shape: Shape::new(out_dims),
            data: out_data,
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let m = mul as f32;
        let a = add as f32;
        let data = self.data.iter().map(|&x| x * m + a).collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn neg(&self) -> Result<Self> { self.affine(-1.0, 0.0) }
    pub fn recip(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| 1.0 / x).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn relu(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn gelu(&self) -> Result<Self> {
        let c1 = (2.0f32 / std::f32::consts::PI).sqrt();
        let data = self.data.iter().map(|&x| {
            let inner = c1 * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn elu(&self, alpha: f64) -> Result<Self> {
        let a = alpha as f32;
        let data = self.data.iter().map(|&x| if x >= 0.0 { x } else { a * (x.exp() - 1.0) }).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn tanh(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.tanh()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn exp(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.exp()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn ln(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.ln()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn log(&self) -> Result<Self> { self.ln() }
    pub fn log10(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.log10()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn sin(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.sin()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn cos(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.cos()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn sqrt(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.sqrt()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn sqr(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x * x).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn abs(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.abs()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn round(&self) -> Result<Self> {
        let data = self.data.iter().map(|&x| x.round()).collect();
        Ok(Self { shape: self.shape.clone(), data, device: self.device, dtype: self.dtype })
    }

    pub fn maximum<T: IntoTensorOrScalar>(&self, other: T) -> Result<Self> {
        let other_t = other.into_tensor(&self.device)?;
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other_t.data, other_t.dims(), |a, b| a.max(b))?;
        Ok(Self { shape, data, device: self.device, dtype: self.dtype })
    }

    pub fn broadcast_maximum<T: IntoTensorOrScalar>(&self, other: T) -> Result<Self> {
        self.maximum(other)
    }

    pub fn minimum<T: IntoTensorOrScalar>(&self, other: T) -> Result<Self> {
        let other_t = other.into_tensor(&self.device)?;
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other_t.data, other_t.dims(), |a, b| a.min(b))?;
        Ok(Self { shape, data, device: self.device, dtype: self.dtype })
    }

    pub fn broadcast_minimum<T: IntoTensorOrScalar>(&self, other: T) -> Result<Self> {
        self.minimum(other)
    }

    pub fn clamp<T1: IntoTensorOrScalar, T2: IntoTensorOrScalar>(&self, min: T1, max: T2) -> Result<Self> {
        self.maximum(min)?.minimum(max)
    }

    pub fn where_cond(&self, on_true: &Tensor, on_false: &Tensor) -> Result<Self> {
        let mut out_data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            let cond = self.data[i] > 0.5;
            let val = if cond { on_true.data[i % on_true.data.len()] } else { on_false.data[i % on_false.data.len()] };
            out_data.push(val);
        }
        Ok(Self { shape: self.shape.clone(), data: out_data, device: self.device, dtype: self.dtype })
    }

    pub fn sum(&self, dim: usize) -> Result<Self> { self.reduce_dim(dim, false, 0.0, |acc, x| acc + x) }
    pub fn sum_keepdim(&self, dim: usize) -> Result<Self> { self.reduce_dim(dim, true, 0.0, |acc, x| acc + x) }
    pub fn mean(&self, dim: usize) -> Result<Self> {
        let sum_t = self.sum(dim)?;
        let count = self.dims()[dim] as f32;
        sum_t.affine((1.0 / count) as f64, 0.0)
    }
    pub fn mean_keepdim(&self, dim: usize) -> Result<Self> {
        let sum_t = self.sum_keepdim(dim)?;
        let count = self.dims()[dim] as f32;
        sum_t.affine((1.0 / count) as f64, 0.0)
    }
    pub fn var(&self, dim: usize) -> Result<Self> {
        let mean = self.mean_keepdim(dim)?;
        let diff = self.broadcast_sub(&mean)?;
        diff.sqr()?.mean(dim)
    }
    pub fn var_keepdim(&self, dim: usize) -> Result<Self> {
        let mean = self.mean_keepdim(dim)?;
        let diff = self.broadcast_sub(&mean)?;
        diff.sqr()?.mean_keepdim(dim)
    }
    pub fn max(&self, dim: usize) -> Result<Self> { self.reduce_dim(dim, false, f32::NEG_INFINITY, |acc, x| acc.max(x)) }
    pub fn max_keepdim(&self, dim: usize) -> Result<Self> { self.reduce_dim(dim, true, f32::NEG_INFINITY, |acc, x| acc.max(x)) }

    pub fn sum_all(&self) -> Result<Self> {
        let sum: f32 = self.data.iter().sum();
        Self::new(sum, &self.device)
    }

    fn reduce_dim<F>(&self, dim: usize, keepdim: bool, init: f32, op: F) -> Result<Self>
    where
        F: Fn(f32, f32) -> f32,
    {
        let dims = self.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("Reduce dim out of bounds".to_string()));
        }

        let mut out_dims = dims.to_vec();
        if keepdim {
            out_dims[dim] = 1;
        } else {
            out_dims.remove(dim);
        }
        if out_dims.is_empty() {
            out_dims.push(1);
        }

        let out_count: usize = out_dims.iter().product();
        let mut out_data = vec![init; out_count];
        let out_strides = compute_strides(&out_dims);

        for (i, &val) in self.data.iter().enumerate() {
            let mut coords = flat_to_coords(i, dims);
            if keepdim {
                coords[dim] = 0;
            } else {
                coords.remove(dim);
            }
            if coords.is_empty() {
                coords.push(0);
            }
            let out_idx = coords_to_flat(&coords, &out_strides);
            out_data[out_idx] = op(out_data[out_idx], val);
        }

        Ok(Self {
            shape: Shape::new(out_dims),
            data: out_data,
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("Pad dim out of bounds".to_string()));
        }

        let mut out_dims = dims.to_vec();
        out_dims[dim] += left + right;
        let out_count: usize = out_dims.iter().product();
        let mut out_data = vec![0.0f32; out_count];
        let out_strides = compute_strides(&out_dims);

        for (i, &val) in self.data.iter().enumerate() {
            let mut coords = flat_to_coords(i, dims);
            coords[dim] += left;
            let out_idx = coords_to_flat(&coords, &out_strides);
            out_data[out_idx] = val;
        }

        Ok(Self {
            shape: Shape::new(out_dims),
            data: out_data,
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn index_select(&self, ids: &Tensor, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("Index select dim out of bounds".to_string()));
        }

        let num_indices = ids.data.len();
        let mut out_dims = dims.to_vec();
        out_dims[dim] = num_indices;

        let mut out_data = Vec::with_capacity(out_dims.iter().product());
        for &id in &ids.data {
            let idx = id as usize;
            let slice = self.narrow(dim, idx, 1)?;
            out_data.extend_from_slice(&slice.data);
        }

        Ok(Self {
            shape: Shape::new(out_dims),
            data: out_data,
            device: self.device,
            dtype: self.dtype,
        })
    }

    pub fn cat<T: AsRef<Tensor>>(tensors: &[T], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(Error::Msg("Empty tensors in cat".to_string()));
        }
        let first = tensors[0].as_ref();
        let rank = first.dims().len();
        if dim >= rank {
            return Err(Error::Msg("Cat dim out of bounds".to_string()));
        }

        let mut total_dim = 0;
        for t in tensors {
            let t_ref = t.as_ref();
            if t_ref.dims().len() != rank {
                return Err(Error::Msg("Cat tensors rank mismatch".to_string()));
            }
            total_dim += t_ref.dims()[dim];
        }

        let mut out_dims = first.dims().to_vec();
        out_dims[dim] = total_dim;
        let mut out_data = vec![0.0f32; out_dims.iter().product()];
        let out_strides = compute_strides(&out_dims);

        let mut current_offset = 0;
        for t in tensors {
            let t_ref = t.as_ref();
            let t_dims = t_ref.dims();
            for (i, &val) in t_ref.data.iter().enumerate() {
                let mut coords = flat_to_coords(i, t_dims);
                coords[dim] += current_offset;
                let out_idx = coords_to_flat(&coords, &out_strides);
                out_data[out_idx] = val;
            }
            current_offset += t_dims[dim];
        }

        Ok(Self {
            shape: Shape::new(out_dims),
            data: out_data,
            device: first.device,
            dtype: first.dtype,
        })
    }

    pub fn stack<T: AsRef<Tensor>>(tensors: &[T], dim: usize) -> Result<Self> {
        let unsqueezed: Result<Vec<Tensor>> = tensors.iter().map(|t| t.as_ref().unsqueeze(dim)).collect();
        Self::cat(&unsqueezed?, dim)
    }

    pub fn conv1d(
        &self,
        weight: &Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (b, c_in, l_in) = self.dims3()?;
        let (c_out, _c_in_per_group, k) = weight.dims3()?;

        let effective_k = (k - 1) * dilation + 1;
        let padded_l = l_in + 2 * padding;
        if padded_l < effective_k {
            return Err(Error::Msg("Padded length is smaller than kernel".to_string()));
        }
        let l_out = (padded_l - effective_k) / stride + 1;

        let c_in_group = c_in / groups;
        let c_out_group = c_out / groups;

        let x_data: Vec<f32> = self.data.clone();
        let w_data: Vec<f32> = weight.data.clone();
        let mut out_data = vec![0.0f32; b * c_out * l_out];

        for batch in 0..b {
            for g in 0..groups {
                for oc in 0..c_out_group {
                    let out_c = g * c_out_group + oc;
                    for ol in 0..l_out {
                        let mut sum = 0.0f32;
                        let in_start = (ol * stride) as isize - padding as isize;

                        for ic in 0..c_in_group {
                            let in_c = g * c_in_group + ic;
                            for ki in 0..k {
                                let in_pos = in_start + (ki * dilation) as isize;
                                if in_pos >= 0 && (in_pos as usize) < l_in {
                                    let x_val = x_data[batch * c_in * l_in + in_c * l_in + (in_pos as usize)];
                                    let w_val = w_data[out_c * (c_in_group * k) + ic * k + ki];
                                    sum += x_val * w_val;
                                }
                            }
                        }

                        let out_idx = batch * (c_out * l_out) + out_c * l_out + ol;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }

        Self::from_vec(out_data, (b, c_out, l_out), &self.device)
    }

    pub fn conv_transpose1d(
        &self,
        weight: &Tensor,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (b, c_in, l_in) = self.dims3()?;
        let (w_in, w_out, k) = weight.dims3()?;
        if w_in != c_in {
            return Err(Error::Msg(format!("Weight in_c {} != input c_in {}", w_in, c_in)));
        }

        let c_out = w_out * groups;
        let effective_k = (k - 1) * dilation + 1;
        let l_out = (l_in - 1) * stride + effective_k + output_padding - 2 * padding;

        let x_data: Vec<f32> = self.data.clone();
        let w_data: Vec<f32> = weight.data.clone();
        let mut out_data = vec![0.0f32; b * c_out * l_out];

        for batch in 0..b {
            for il in 0..l_in {
                for ic in 0..c_in {
                    let in_val = x_data[batch * (c_in * l_in) + ic * l_in + il];
                    if in_val == 0.0 {
                        continue;
                    }
                    for oc in 0..c_out {
                        for ki in 0..k {
                            let out_pos = (il * stride + ki * dilation) as isize - padding as isize;
                            if out_pos >= 0 && (out_pos as usize) < l_out {
                                let w_val = w_data[ic * (c_out * k) + oc * k + ki];
                                let out_idx = batch * (c_out * l_out) + oc * l_out + (out_pos as usize);
                                out_data[out_idx] += in_val * w_val;
                            }
                        }
                    }
                }
            }
        }

        Self::from_vec(out_data, (b, c_out, l_out), &self.device)
    }

    pub fn ge(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| if a >= b { 1.0 } else { 0.0 })?;
        Ok(Self { shape, data, device: self.device, dtype: DType::U8 })
    }

    pub fn le(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| if a <= b { 1.0 } else { 0.0 })?;
        Ok(Self { shape, data, device: self.device, dtype: DType::U8 })
    }

    pub fn gt(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| if a > b { 1.0 } else { 0.0 })?;
        Ok(Self { shape, data, device: self.device, dtype: DType::U8 })
    }

    pub fn lt(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| if a < b { 1.0 } else { 0.0 })?;
        Ok(Self { shape, data, device: self.device, dtype: DType::U8 })
    }

    pub fn eq(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 })?;
        Ok(Self { shape, data, device: self.device, dtype: DType::U8 })
    }

    pub fn ne(&self, other: &Tensor) -> Result<Self> {
        let (data, shape) = broadcast_binary_op(&self.data, self.dims(), &other.data, other.dims(), |a, b| if (a - b).abs() >= 1e-6 { 1.0 } else { 0.0 })?;
        Ok(Self { shape, data, device: self.device, dtype: DType::U8 })
    }

    pub fn to_scalar<T: FromF32>(&self) -> Result<T> {
        if self.data.is_empty() {
            return Err(Error::Msg("Empty tensor has no scalar value".to_string()));
        }
        Ok(T::from_f32(self.data[0]))
    }

    pub fn to_vec1<T: FromF32>(&self) -> Result<Vec<T>> {
        if self.dims().len() != 1 {
            return Err(Error::Msg(format!("to_vec1 expected 1D, got {:?}", self.dims())));
        }
        Ok(self.data.iter().map(|&x| T::from_f32(x)).collect())
    }

    pub fn to_vec2<T: FromF32>(&self) -> Result<Vec<Vec<T>>> {
        let dims = self.dims();
        if dims.len() != 2 {
            return Err(Error::Msg(format!("to_vec2 expected 2D, got {:?}", dims)));
        }
        let (d0, d1) = (dims[0], dims[1]);
        let mut out = Vec::with_capacity(d0);
        for i in 0..d0 {
            let row = self.data[i * d1..(i + 1) * d1]
                .iter()
                .map(|&x| T::from_f32(x))
                .collect();
            out.push(row);
        }
        Ok(out)
    }

    pub fn to_vec3<T: FromF32>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        let dims = self.dims();
        if dims.len() != 3 {
            return Err(Error::Msg(format!("to_vec3 expected 3D, got {:?}", dims)));
        }
        let (d0, d1, d2) = (dims[0], dims[1], dims[2]);
        let mut out = Vec::with_capacity(d0);
        for i in 0..d0 {
            let mut matrix = Vec::with_capacity(d1);
            for j in 0..d1 {
                let start = (i * d1 + j) * d2;
                let row = self.data[start..start + d2]
                    .iter()
                    .map(|&x| T::from_f32(x))
                    .collect();
                matrix.push(row);
            }
            out.push(matrix);
        }
        Ok(out)
    }
}

pub trait IntoF32 {
    fn into_f32(self) -> f32;
}
impl IntoF32 for f32 { fn into_f32(self) -> f32 { self } }
impl IntoF32 for f64 { fn into_f32(self) -> f32 { self as f32 } }
impl IntoF32 for u32 { fn into_f32(self) -> f32 { self as f32 } }
impl IntoF32 for i32 { fn into_f32(self) -> f32 { self as f32 } }
impl IntoF32 for i64 { fn into_f32(self) -> f32 { self as f32 } }
impl IntoF32 for usize { fn into_f32(self) -> f32 { self as f32 } }
impl IntoF32 for u8 { fn into_f32(self) -> f32 { self as f32 } }

pub trait FromF32 {
    fn from_f32(val: f32) -> Self;
}
impl FromF32 for f32 { fn from_f32(val: f32) -> Self { val } }
impl FromF32 for f64 { fn from_f32(val: f32) -> Self { val as f64 } }
impl FromF32 for u32 { fn from_f32(val: f32) -> Self { val as u32 } }
impl FromF32 for i32 { fn from_f32(val: f32) -> Self { val as i32 } }
impl FromF32 for i64 { fn from_f32(val: f32) -> Self { val as i64 } }
impl FromF32 for usize { fn from_f32(val: f32) -> Self { val as usize } }
impl FromF32 for u8 { fn from_f32(val: f32) -> Self { val as u8 } }

pub trait IntoTensorOrScalar {
    fn into_tensor(self, device: &Device) -> Result<Tensor>;
}

impl IntoTensorOrScalar for &Tensor {
    fn into_tensor(self, _device: &Device) -> Result<Tensor> { Ok(self.clone()) }
}

impl IntoTensorOrScalar for Tensor {
    fn into_tensor(self, _device: &Device) -> Result<Tensor> { Ok(self) }
}

impl IntoTensorOrScalar for f32 {
    fn into_tensor(self, device: &Device) -> Result<Tensor> { Tensor::new(self, device) }
}

impl IntoTensorOrScalar for f64 {
    fn into_tensor(self, device: &Device) -> Result<Tensor> { Tensor::new(self, device) }
}

impl IntoTensorOrScalar for u32 {
    fn into_tensor(self, device: &Device) -> Result<Tensor> { Tensor::new(self, device) }
}

impl IntoTensorOrScalar for i64 {
    fn into_tensor(self, device: &Device) -> Result<Tensor> { Tensor::new(self, device) }
}

impl IntoTensorOrScalar for usize {
    fn into_tensor(self, device: &Device) -> Result<Tensor> { Tensor::new(self, device) }
}

pub trait IntoTensorData {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType);
}

impl IntoTensorData for f32 {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        (vec![self], Shape::new(vec![]), DType::F32)
    }
}

impl IntoTensorData for f64 {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        (vec![self as f32], Shape::new(vec![]), DType::F64)
    }
}

impl IntoTensorData for u32 {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        (vec![self as f32], Shape::new(vec![]), DType::U32)
    }
}

impl IntoTensorData for i64 {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        (vec![self as f32], Shape::new(vec![]), DType::I64)
    }
}

impl IntoTensorData for usize {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        (vec![self as f32], Shape::new(vec![]), DType::U32)
    }
}

impl IntoTensorData for Vec<f32> {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self, Shape::new(vec![len]), DType::F32)
    }
}

impl IntoTensorData for &[f32] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.to_vec(), Shape::new(vec![len]), DType::F32)
    }
}

impl IntoTensorData for Vec<f64> {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.into_iter().map(|x| x as f32).collect(), Shape::new(vec![len]), DType::F64)
    }
}

impl IntoTensorData for &[f64] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.iter().map(|&x| x as f32).collect(), Shape::new(vec![len]), DType::F64)
    }
}

impl IntoTensorData for Vec<u32> {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.into_iter().map(|x| x as f32).collect(), Shape::new(vec![len]), DType::U32)
    }
}

impl IntoTensorData for &[u32] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.iter().map(|&x| x as f32).collect(), Shape::new(vec![len]), DType::U32)
    }
}

impl IntoTensorData for Vec<i64> {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.into_iter().map(|x| x as f32).collect(), Shape::new(vec![len]), DType::I64)
    }
}

impl IntoTensorData for &[i64] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.iter().map(|&x| x as f32).collect(), Shape::new(vec![len]), DType::I64)
    }
}

impl IntoTensorData for Vec<usize> {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.into_iter().map(|x| x as f32).collect(), Shape::new(vec![len]), DType::U32)
    }
}

impl IntoTensorData for &[usize] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let len = self.len();
        (self.iter().map(|&x| x as f32).collect(), Shape::new(vec![len]), DType::U32)
    }
}

impl IntoTensorData for Vec<Vec<f32>> {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let rows = self.len();
        let cols = if rows > 0 { self[0].len() } else { 0 };
        let flat = self.into_iter().flatten().collect();
        (flat, Shape::new(vec![rows, cols]), DType::F32)
    }
}

impl IntoTensorData for &[&[f32]] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let rows = self.len();
        let cols = if rows > 0 { self[0].len() } else { 0 };
        let flat = self.iter().flat_map(|r| r.iter().copied()).collect();
        (flat, Shape::new(vec![rows, cols]), DType::F32)
    }
}

impl<T: IntoF32 + Clone, const N: usize> IntoTensorData for &[T; N] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let f32_data: Vec<f32> = self.iter().map(|x| x.clone().into_f32()).collect();
        (f32_data, Shape::new(vec![N]), DType::F32)
    }
}

impl<T: IntoF32 + Clone, const N: usize> IntoTensorData for [T; N] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let f32_data: Vec<f32> = self.into_iter().map(|x| x.into_f32()).collect();
        (f32_data, Shape::new(vec![N]), DType::F32)
    }
}

impl<T: IntoF32 + Clone, const N1: usize, const N2: usize> IntoTensorData for &[[T; N2]; N1] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let mut flat = Vec::with_capacity(N1 * N2);
        for row in self {
            for item in row {
                flat.push(item.clone().into_f32());
            }
        }
        (flat, Shape::new(vec![N1, N2]), DType::F32)
    }
}

impl<T: IntoF32 + Clone, const N1: usize, const N2: usize, const N3: usize> IntoTensorData for &[[[T; N3]; N2]; N1] {
    fn into_tensor_data(self) -> (Vec<f32>, Shape, DType) {
        let mut flat = Vec::with_capacity(N1 * N2 * N3);
        for mat in self {
            for row in mat {
                for item in row {
                    flat.push(item.clone().into_f32());
                }
            }
        }
        (flat, Shape::new(vec![N1, N2, N3]), DType::F32)
    }
}

// Operators
impl Add<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_add(rhs) }
}
impl Add<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_add(&rhs) }
}
impl Add<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_add(rhs) }
}
impl Add<Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_add(&rhs) }
}
impl Add<f64> for Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f64) -> Result<Tensor> { self.affine(1.0, rhs) }
}
impl Add<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f64) -> Result<Tensor> { self.affine(1.0, rhs) }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_sub(rhs) }
}
impl Sub<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_sub(&rhs) }
}
impl Sub<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_sub(rhs) }
}
impl Sub<Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_sub(&rhs) }
}
impl Sub<f64> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { self.affine(1.0, -rhs) }
}
impl Sub<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { self.affine(1.0, -rhs) }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_mul(rhs) }
}
impl Mul<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_mul(&rhs) }
}
impl Mul<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_mul(rhs) }
}
impl Mul<Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_mul(&rhs) }
}
impl Mul<f64> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { self.affine(rhs, 0.0) }
}
impl Mul<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { self.affine(rhs, 0.0) }
}

impl Div<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_div(rhs) }
}
impl Div<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_div(&rhs) }
}
impl Div<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: &Tensor) -> Result<Tensor> { self.broadcast_div(rhs) }
}
impl Div<Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: Tensor) -> Result<Tensor> { self.broadcast_div(&rhs) }
}
impl Div<f64> for Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { self.affine(1.0 / rhs, 0.0) }
}
impl Div<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { self.affine(1.0 / rhs, 0.0) }
}

impl Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { self.affine(-1.0, 0.0) }
}
impl Neg for Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { self.affine(-1.0, 0.0) }
}
