use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn elem_count(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

pub trait ShapeDim {
    fn to_dim(self) -> Option<usize>;
}

impl ShapeDim for usize {
    fn to_dim(self) -> Option<usize> {
        Some(self)
    }
}

impl ShapeDim for () {
    fn to_dim(self) -> Option<usize> {
        None
    }
}

pub trait IntoShape {
    fn into_shape(self) -> Shape;
}

pub trait IntoShapeWithInferred {
    fn into_shape_with_inferred(self, total_elements: usize) -> Result<Shape>;
}

// 1D
impl<D0: ShapeDim> IntoShapeWithInferred for (D0,) {
    fn into_shape_with_inferred(self, total_elements: usize) -> Result<Shape> {
        let d0 = self.0.to_dim().unwrap_or(total_elements);
        Ok(Shape::new(vec![d0]))
    }
}

// 2D
impl<D0: ShapeDim, D1: ShapeDim> IntoShapeWithInferred for (D0, D1) {
    fn into_shape_with_inferred(self, total_elements: usize) -> Result<Shape> {
        let dims_opt = [self.0.to_dim(), self.1.to_dim()];
        resolve_inferred(&dims_opt, total_elements)
    }
}

// 3D
impl<D0: ShapeDim, D1: ShapeDim, D2: ShapeDim> IntoShapeWithInferred for (D0, D1, D2) {
    fn into_shape_with_inferred(self, total_elements: usize) -> Result<Shape> {
        let dims_opt = [self.0.to_dim(), self.1.to_dim(), self.2.to_dim()];
        resolve_inferred(&dims_opt, total_elements)
    }
}

// 4D
impl<D0: ShapeDim, D1: ShapeDim, D2: ShapeDim, D3: ShapeDim> IntoShapeWithInferred for (D0, D1, D2, D3) {
    fn into_shape_with_inferred(self, total_elements: usize) -> Result<Shape> {
        let dims_opt = [self.0.to_dim(), self.1.to_dim(), self.2.to_dim(), self.3.to_dim()];
        resolve_inferred(&dims_opt, total_elements)
    }
}

fn resolve_inferred(dims_opt: &[Option<usize>], total_elements: usize) -> Result<Shape> {
    let mut known_prod = 1;
    let mut inferred_idx = None;
    for (i, &opt) in dims_opt.iter().enumerate() {
        match opt {
            Some(v) => known_prod *= v,
            None => {
                if inferred_idx.is_some() {
                    return Err(Error::Msg("Only one inferred dimension allowed".to_string()));
                }
                inferred_idx = Some(i);
            }
        }
    }

    let mut out_dims = Vec::with_capacity(dims_opt.len());
    for (i, &opt) in dims_opt.iter().enumerate() {
        if Some(i) == inferred_idx {
            if known_prod == 0 || total_elements % known_prod != 0 {
                return Err(Error::Msg(format!(
                    "Cannot infer dim: total elements {} not divisible by known product {}",
                    total_elements, known_prod
                )));
            }
            out_dims.push(total_elements / known_prod);
        } else {
            out_dims.push(opt.unwrap());
        }
    }
    Ok(Shape::new(out_dims))
}

impl IntoShape for usize {
    fn into_shape(self) -> Shape {
        Shape::new(vec![self])
    }
}

impl IntoShape for (usize,) {
    fn into_shape(self) -> Shape {
        Shape::new(vec![self.0])
    }
}

impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Shape {
        Shape::new(vec![self.0, self.1])
    }
}

impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Shape {
        Shape::new(vec![self.0, self.1, self.2])
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Shape {
        Shape::new(vec![self.0, self.1, self.2, self.3])
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Shape {
        Shape::new(self.to_vec())
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Shape {
        Shape::new(self)
    }
}

impl IntoShape for &Shape {
    fn into_shape(self) -> Shape {
        self.clone()
    }
}

impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}

impl IntoShapeWithInferred for usize {
    fn into_shape_with_inferred(self, _total_elements: usize) -> Result<Shape> {
        Ok(Shape::new(vec![self]))
    }
}

impl IntoShapeWithInferred for &[usize] {
    fn into_shape_with_inferred(self, _total_elements: usize) -> Result<Shape> {
        Ok(Shape::new(self.to_vec()))
    }
}

impl IntoShapeWithInferred for Vec<usize> {
    fn into_shape_with_inferred(self, _total_elements: usize) -> Result<Shape> {
        Ok(Shape::new(self))
    }
}

impl IntoShapeWithInferred for &Shape {
    fn into_shape_with_inferred(self, _total_elements: usize) -> Result<Shape> {
        Ok(self.clone())
    }
}

impl IntoShapeWithInferred for Shape {
    fn into_shape_with_inferred(self, _total_elements: usize) -> Result<Shape> {
        Ok(self)
    }
}

