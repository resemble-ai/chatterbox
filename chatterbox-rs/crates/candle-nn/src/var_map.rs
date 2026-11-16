use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use candle_core::{DType, Device, IntoShape, Result, Tensor};

#[derive(Clone, Debug)]
pub enum Init {
    Const(f64),
    Randn { mean: f64, stdev: f64 },
}

#[derive(Clone, Default)]
pub struct VarMap {
    vars: Rc<RefCell<HashMap<String, Tensor>>>,
}

impl VarMap {
    pub fn new() -> Self {
        Self {
            vars: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    pub fn set(&self, name: &str, tensor: Tensor) {
        self.vars.borrow_mut().insert(name.to_string(), tensor);
    }

    pub fn get(&self, name: &str) -> Option<Tensor> {
        self.vars.borrow().get(name).cloned()
    }
}

#[derive(Clone)]
pub struct VarBuilder {
    prefix: String,
    varmap: VarMap,
    dtype: DType,
    device: Device,
}

impl VarBuilder {
    pub fn from_varmap(varmap: &VarMap, dtype: DType, device: &Device) -> Self {
        Self {
            prefix: String::new(),
            varmap: varmap.clone(),
            dtype,
            device: *device,
        }
    }

    pub fn zeros(dtype: DType, device: &Device) -> Self {
        Self {
            prefix: String::new(),
            varmap: VarMap::new(),
            dtype,
            device: *device,
        }
    }

    pub fn pp<S: AsRef<str>>(&self, sub: S) -> Self {
        let sub_str = sub.as_ref();
        let new_prefix = if self.prefix.is_empty() {
            sub_str.to_string()
        } else {
            format!("{}.{}", self.prefix, sub_str)
        };
        Self {
            prefix: new_prefix,
            varmap: self.varmap.clone(),
            dtype: self.dtype,
            device: self.device,
        }
    }

    pub fn get<S: IntoShape>(&self, shape: S, name: &str) -> Result<Tensor> {
        self.get_with_hints(shape, name, Init::Const(0.0))
    }

    pub fn get_with_hints<S: IntoShape>(&self, shape: S, name: &str, init: Init) -> Result<Tensor> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };

        if let Some(t) = self.varmap.get(&full_name) {
            return Ok(t);
        }

        let s = shape.into_shape();
        let tensor = match init {
            Init::Const(c) => {
                let mut t = Tensor::zeros(s, self.dtype, &self.device)?;
                if c != 0.0 {
                    t = t.affine(1.0, c)?;
                }
                t
            }
            Init::Randn { mean, stdev } => Tensor::randn(mean as f32, stdev as f32, s, &self.device)?,
        };

        self.varmap.set(&full_name, tensor.clone());
        Ok(tensor)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
