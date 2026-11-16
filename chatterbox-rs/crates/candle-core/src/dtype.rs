#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    U8,
    U32,
    I64,
    F32,
    F64,
}

impl Default for DType {
    fn default() -> Self {
        DType::F32
    }
}
