pub mod device;
pub mod dtype;
pub mod error;
pub mod module;
pub mod ops;
pub mod shape;
pub mod tensor;

pub use device::Device;
pub use dtype::DType;
pub use error::{Error, Result};
pub use module::{Module, ModuleT};
pub use shape::{IntoShape, IntoShapeWithInferred, Shape};
pub use tensor::Tensor;

pub mod utils {
    pub fn cuda_is_available() -> bool {
        false
    }
    pub fn metal_is_available() -> bool {
        false
    }
}
