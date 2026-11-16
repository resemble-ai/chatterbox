use candle_core::{Error, Module, Result, Tensor};
use crate::var_map::{Init, VarBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv1dConfig) -> Self {
        Self { weight, bias, config }
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_in, l_in) = x.dims3()?;
        let (c_out, _c_in_per_group, k) = self.weight.dims3()?;

        let padding = self.config.padding;
        let stride = self.config.stride;
        let dilation = self.config.dilation;
        let groups = self.config.groups;

        let effective_k = (k - 1) * dilation + 1;
        let padded_l = l_in + 2 * padding;
        if padded_l < effective_k {
            return Err(Error::Msg("Padded length is smaller than kernel".to_string()));
        }
        let l_out = (padded_l - effective_k) / stride + 1;

        let c_in_group = c_in / groups;
        let c_out_group = c_out / groups;

        let x_data: Vec<f32> = x.to_vec3::<f32>()?.into_iter().flatten().flatten().collect();
        let w_data: Vec<f32> = self.weight.to_vec3::<f32>()?.into_iter().flatten().flatten().collect();

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

        let mut out_tensor = Tensor::from_vec(out_data, (b, c_out, l_out), x.device())?;
        if let Some(ref bias) = self.bias {
            let b_reshaped = bias.reshape((1, c_out, 1))?;
            out_tensor = out_tensor.broadcast_add(&b_reshaped)?;
        }
        Ok(out_tensor)
    }
}

pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let bound = 1.0 / ((in_channels * kernel_size) as f64).sqrt();
    let weight = vb.get_with_hints(
        (out_channels, in_channels / config.groups, kernel_size),
        "weight",
        Init::Randn { mean: 0.0, stdev: bound },
    )?;
    let bias = vb.get_with_hints((out_channels,), "bias", Init::Const(0.0))?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv2dConfig) -> Self {
        Self { weight, bias, config }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_in, h_in, w_in) = x.dims4()?;
        let (c_out, _c_in_per_group, kh, kw) = self.weight.dims4()?;

        let pad = self.config.padding;
        let stride = self.config.stride;
        let h_out = (h_in + 2 * pad - kh) / stride + 1;
        let w_out = (w_in + 2 * pad - kw) / stride + 1;

        let x_data: Vec<f32> = x.flatten_all()?.to_vec1::<f32>()?;
        let w_data: Vec<f32> = self.weight.flatten_all()?.to_vec1::<f32>()?;

        let mut out_data = vec![0.0f32; b * c_out * h_out * w_out];

        for batch in 0..b {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        let h_start = (oh * stride) as isize - pad as isize;
                        let w_start = (ow * stride) as isize - pad as isize;

                        for ic in 0..c_in {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let hp = h_start + ki as isize;
                                    let wp = w_start + kj as isize;
                                    if hp >= 0 && (hp as usize) < h_in && wp >= 0 && (wp as usize) < w_in {
                                        let x_val = x_data[batch * (c_in * h_in * w_in) + ic * (h_in * w_in) + (hp as usize) * w_in + (wp as usize)];
                                        let w_val = w_data[oc * (c_in * kh * kw) + ic * (kh * kw) + ki * kw + kj];
                                        sum += x_val * w_val;
                                    }
                                }
                            }
                        }

                        let out_idx = batch * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }

        let mut out_tensor = Tensor::from_vec(out_data, (b, c_out, h_out, w_out), x.device())?;
        if let Some(ref bias) = self.bias {
            let b_reshaped = bias.reshape((1, c_out, 1, 1))?;
            out_tensor = out_tensor.broadcast_add(&b_reshaped)?;
        }
        Ok(out_tensor)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv2dConfig,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let bound = 1.0 / ((in_channels * kernel_size * kernel_size) as f64).sqrt();
    let weight = vb.get_with_hints(
        (out_channels, in_channels / config.groups, kernel_size, kernel_size),
        "weight",
        Init::Randn { mean: 0.0, stdev: bound },
    )?;
    let bias = vb.get_with_hints((out_channels,), "bias", Init::Const(0.0))?;
    Ok(Conv2d::new(weight, Some(bias), config))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose1dConfig {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for ConvTranspose1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTranspose1dConfig,
}

impl ConvTranspose1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose1dConfig) -> Self {
        Self { weight, bias, config }
    }
}

impl Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_in, l_in) = x.dims3()?;
        let (w_in, w_out, k) = self.weight.dims3()?;
        if w_in != c_in {
            return Err(Error::Msg(format!("Weight in_c {} != input c_in {}", w_in, c_in)));
        }

        let stride = self.config.stride;
        let padding = self.config.padding;
        let dilation = self.config.dilation;
        let out_padding = self.config.output_padding;
        let c_out = w_out * self.config.groups;

        let effective_k = (k - 1) * dilation + 1;
        let l_out = (l_in - 1) * stride + effective_k + out_padding - 2 * padding;

        let x_data: Vec<f32> = x.flatten_all()?.to_vec1::<f32>()?;
        let w_data: Vec<f32> = self.weight.flatten_all()?.to_vec1::<f32>()?;

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

        let mut out_tensor = Tensor::from_vec(out_data, (b, c_out, l_out), x.device())?;
        if let Some(ref bias) = self.bias {
            let b_reshaped = bias.reshape((1, c_out, 1))?;
            out_tensor = out_tensor.broadcast_add(&b_reshaped)?;
        }
        Ok(out_tensor)
    }
}

pub fn conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let bound = 1.0 / ((in_channels * kernel_size) as f64).sqrt();
    let weight = vb.get_with_hints(
        (in_channels, out_channels / config.groups, kernel_size),
        "weight",
        Init::Randn { mean: 0.0, stdev: bound },
    )?;
    let bias = vb.get_with_hints((out_channels,), "bias", Init::Const(0.0))?;
    Ok(ConvTranspose1d::new(weight, Some(bias), config))
}
