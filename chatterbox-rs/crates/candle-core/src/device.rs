use crate::error::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl Device {
    pub fn new_cuda(_id: usize) -> Result<Self> {
        Ok(Device::Cpu)
    }

    pub fn new_metal(_id: usize) -> Result<Self> {
        Ok(Device::Cpu)
    }
}
