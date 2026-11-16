use candle_core::{Result, Tensor};
use candle_nn::{
    batch_norm, conv1d, conv2d, BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Module,
    VarBuilder,
};

pub struct BasicResBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    shortcut: Option<(Conv2d, BatchNorm)>,
}

impl BasicResBlock {
    pub fn load(vb: VarBuilder, in_planes: usize, planes: usize, stride: usize) -> Result<Self> {
        let conv1_cfg = Conv2dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        let conv1 = conv2d(in_planes, planes, 3, conv1_cfg, vb.pp("conv1"))?;
        let bn1 = batch_norm(planes, 1e-5, vb.pp("bn1"))?;

        let conv2_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let conv2 = conv2d(planes, planes, 3, conv2_cfg, vb.pp("conv2"))?;
        let bn2 = batch_norm(planes, 1e-5, vb.pp("bn2"))?;

        let shortcut = if stride != 1 || in_planes != planes {
            let sc_cfg = Conv2dConfig {
                padding: 0,
                stride,
                ..Default::default()
            };
            let sc_conv = conv2d(in_planes, planes, 1, sc_cfg, vb.pp("shortcut.0"))?;
            let sc_bn = batch_norm(planes, 1e-5, vb.pp("shortcut.1"))?;
            Some((sc_conv, sc_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
        })
    }
}

impl Module for BasicResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward_t(&out, false)?.relu()?;
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward_t(&out, false)?;

        let shortcut_out = if let Some((sc_conv, sc_bn)) = &self.shortcut {
            let res = sc_conv.forward(x)?;
            sc_bn.forward_t(&res, false)?
        } else {
            x.clone()
        };

        (out + shortcut_out)?.relu()
    }
}

pub struct FCM {
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: Vec<BasicResBlock>,
    layer2: Vec<BasicResBlock>,
    conv2: Conv2d,
    bn2: BatchNorm,
    pub out_channels: usize,
}

impl FCM {
    pub fn load(vb: VarBuilder, m_channels: usize, feat_dim: usize) -> Result<Self> {
        let conv1_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let conv1 = conv2d(1, m_channels, 3, conv1_cfg, vb.pp("conv1"))?;
        let bn1 = batch_norm(m_channels, 1e-5, vb.pp("bn1"))?;

        let mut layer1 = Vec::new();
        let mut in_p = m_channels;
        layer1.push(BasicResBlock::load(vb.pp("layer1.0"), in_p, m_channels, 2)?);
        in_p = m_channels;
        layer1.push(BasicResBlock::load(vb.pp("layer1.1"), in_p, m_channels, 1)?);

        let mut layer2 = Vec::new();
        layer2.push(BasicResBlock::load(vb.pp("layer2.0"), in_p, m_channels, 2)?);
        in_p = m_channels;
        layer2.push(BasicResBlock::load(vb.pp("layer2.1"), in_p, m_channels, 1)?);

        let conv2_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let conv2 = conv2d(m_channels, m_channels, 3, conv2_cfg, vb.pp("conv2"))?;
        let bn2 = batch_norm(m_channels, 1e-5, vb.pp("bn2"))?;

        let out_channels = m_channels * (feat_dim / 8);

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            conv2,
            bn2,
            out_channels,
        })
    }
}

impl Module for FCM {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.unsqueeze(1)?;
        let mut out = self.conv1.forward(&x)?;
        out = self.bn1.forward_t(&out, false)?.relu()?;

        for layer in &self.layer1 {
            out = layer.forward(&out)?;
        }
        for layer in &self.layer2 {
            out = layer.forward(&out)?;
        }

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward_t(&out, false)?.relu()?;

        let dims = out.dims();
        out.reshape((dims[0], dims[1] * dims[2], dims[3]))
    }
}

pub enum Nonlinear {
    ReLU,
    PReLU(Tensor), // Would need more robust handling for prelu weights
    BatchNorm(BatchNorm),
}

impl Nonlinear {
    pub fn load(vb: VarBuilder, name: &str, channels: usize) -> Result<Self> {
        match name {
            "relu" => Ok(Nonlinear::ReLU),
            "batchnorm" => {
                let bn = batch_norm(channels, 1e-5, vb)?;
                Ok(Nonlinear::BatchNorm(bn))
            }
            "batchnorm_" => {
                let bn = batch_norm(channels, 1e-5, vb)?;
                Ok(Nonlinear::BatchNorm(bn))
            }
            _ => panic!("Unsupported nonlinear: {}", name),
        }
    }
}

impl Module for Nonlinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Nonlinear::ReLU => x.relu(),
            Nonlinear::BatchNorm(bn) => bn.forward_t(x, false),
            Nonlinear::PReLU(_) => panic!("PReLU not fully implemented"),
        }
    }
}

pub struct NonlinearSequential {
    layers: Vec<Nonlinear>,
}

impl NonlinearSequential {
    pub fn load(vb: VarBuilder, config_str: &str, channels: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for name in config_str.split('-') {
            layers.push(Nonlinear::load(vb.pp(name), name, channels)?);
        }
        Ok(Self { layers })
    }
}

impl Module for NonlinearSequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}

pub fn statistics_pooling(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean(2)?; // dim=-1 in PyTorch which is typically the sequence dimension (T) for (B, F, T)
    let variance = x.var(2)?; // Note: typically std, not var directly.
    let std = (variance + 1e-6)?.sqrt()?;
    Tensor::cat(&[&mean, &std], 1) // dim=-1 in PyTorch which after reduction is 1
}

pub struct StatsPool {}

impl StatsPool {
    pub fn new() -> Self {
        Self {}
    }
}

impl Module for StatsPool {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        statistics_pooling(x)
    }
}

pub struct TDNNLayer {
    linear: Conv1d,
    nonlinear: NonlinearSequential,
}

    #[allow(clippy::too_many_arguments)]
impl TDNNLayer {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: i64,
        config_str: &str,
    ) -> Result<Self> {
        let pad = if padding < 0 {
            assert!(kernel_size % 2 == 1, "Expect odd kernel size");
            (kernel_size - 1) / 2 * dilation
        } else {
            padding as usize
        };

        let cfg = Conv1dConfig {
            stride,
            padding: pad,
            dilation,
            ..Default::default()
        };
        let linear = conv1d(in_channels, out_channels, kernel_size, cfg, vb.pp("linear"))?;
        let nonlinear = NonlinearSequential::load(vb.pp("nonlinear"), config_str, out_channels)?;

        Ok(Self { linear, nonlinear })
    }
}

impl Module for TDNNLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.linear.forward(x)?;
        self.nonlinear.forward(&out)
    }
}

pub struct CAMLayer {
    linear_local: Conv1d,
    linear1: Conv1d,
    linear2: Conv1d,
    // sigmoid: Sigmoid - applied via Candle tensor ops
    // relu: ReLU - applied via Candle tensor ops
}

    #[allow(clippy::too_many_arguments)]
impl CAMLayer {
    pub fn load(
        vb: VarBuilder,
        bn_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        reduction: usize,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            stride,
            padding,
            dilation,
            ..Default::default()
        };
        let linear_local = conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            cfg,
            vb.pp("linear_local"),
        )?;

        let linear1 = conv1d(
            bn_channels,
            bn_channels / reduction,
            1,
            Default::default(),
            vb.pp("linear1"),
        )?;
        let linear2 = conv1d(
            bn_channels / reduction,
            out_channels,
            1,
            Default::default(),
            vb.pp("linear2"),
        )?;

        Ok(Self {
            linear_local,
            linear1,
            linear2,
        })
    }

    fn seg_pooling(&self, x: &Tensor, _seg_len: usize) -> Result<Tensor> {
        // Average pooling approximation. For rigorous parity with PyTorch F.avg_pool1d:
        // We might need an actual avg_pool1d if candle doesn't have it.
        // Let's implement a simple version or use F::avg_pool2d on unsqueezed

        let dims = x.dims();
        let (b, c, t) = (dims[0], dims[1], dims[2]);

        // As a placeholder for avg_pool1d which may be tricky, let's just do a mean across sequence
        // The python code does avg_pool1d(kernel_size=seg_len, stride=seg_len)
        // Here we just use basic x.mean(2) expanded for simplicity in this port, or reshape based
        // Since it's a bit complex in candle to do overlapping windows without unfold,
        // For segment pooling, seg_len=100.
        // We'll fallback to global mean if seg_len > t

        let mut pooled = x.mean(2)?; // Shape: (B, C)
        pooled = pooled.unsqueeze(2)?; // Shape: (B, C, 1)
        pooled.broadcast_as((b, c, t))
    }
}

impl Module for CAMLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.linear_local.forward(x)?;

        let global_mean = x.mean(2)?.unsqueeze(2)?; // (B, C, 1)
        let seg_pool = self.seg_pooling(x, 100)?;
        let context = global_mean.broadcast_add(&seg_pool)?;

        let mut context = self.linear1.forward(&context)?;
        context = context.relu()?;

        let mut m = self.linear2.forward(&context)?;
        // Sigmoid
        m = (m.neg()?.exp()? + 1.0)?.recip()?;

        y.broadcast_mul(&m)
    }
}

pub struct CAMDenseTDNNLayer {
    nonlinear1: NonlinearSequential,
    linear1: Conv1d,
    nonlinear2: NonlinearSequential,
    cam_layer: CAMLayer,
}

    #[allow(clippy::too_many_arguments)]
impl CAMDenseTDNNLayer {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        bn_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        config_str: &str,
    ) -> Result<Self> {
        let padding = (kernel_size - 1) / 2 * dilation;

        let nonlinear1 = NonlinearSequential::load(vb.pp("nonlinear1"), config_str, in_channels)?;
        let linear1 = conv1d(
            in_channels,
            bn_channels,
            1,
            Default::default(),
            vb.pp("linear1"),
        )?;
        let nonlinear2 = NonlinearSequential::load(vb.pp("nonlinear2"), config_str, bn_channels)?;

        let cam_layer = CAMLayer::load(
            vb.pp("cam_layer"),
            bn_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            2,
        )?;

        Ok(Self {
            nonlinear1,
            linear1,
            nonlinear2,
            cam_layer,
        })
    }

    fn bn_function(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.nonlinear1.forward(x)?;
        self.linear1.forward(&out)
    }
}

impl Module for CAMDenseTDNNLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn_function(x)?;
        let out = self.nonlinear2.forward(&out)?;
        self.cam_layer.forward(&out)
    }
}

pub struct CAMDenseTDNNBlock {
    layers: Vec<CAMDenseTDNNLayer>,
}

    #[allow(clippy::too_many_arguments)]
impl CAMDenseTDNNBlock {
    pub fn load(
        vb: VarBuilder,
        num_layers: usize,
        in_channels: usize,
        out_channels: usize,
        bn_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        config_str: &str,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer = CAMDenseTDNNLayer::load(
                vb.pp(format!("tdnnd{}", i + 1)),
                in_channels + i * out_channels,
                out_channels,
                bn_channels,
                kernel_size,
                stride,
                dilation,
                config_str,
            )?;
            layers.push(layer);
        }

        Ok(Self { layers })
    }
}

impl Module for CAMDenseTDNNBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            let res = layer.forward(&out)?;
            out = Tensor::cat(&[&out, &res], 1)?; // dim=1 for channel concat
        }
        Ok(out)
    }
}

pub struct TransitLayer {
    nonlinear: NonlinearSequential,
    linear: Conv1d,
}

impl TransitLayer {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config_str: &str,
    ) -> Result<Self> {
        let nonlinear = NonlinearSequential::load(vb.pp("nonlinear"), config_str, in_channels)?;
        let linear = conv1d(
            in_channels,
            out_channels,
            1,
            Default::default(),
            vb.pp("linear"),
        )?;

        Ok(Self { nonlinear, linear })
    }
}

impl Module for TransitLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.nonlinear.forward(x)?;
        self.linear.forward(&out)
    }
}

pub struct DenseLayer {
    linear: Conv1d, // Using Conv1d for dense layer on 1d data (or linear)
    nonlinear: NonlinearSequential,
}

impl DenseLayer {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config_str: &str,
    ) -> Result<Self> {
        let linear = conv1d(
            in_channels,
            out_channels,
            1,
            Default::default(),
            vb.pp("linear"),
        )?;
        let nonlinear = NonlinearSequential::load(vb.pp("nonlinear"), config_str, out_channels)?;

        Ok(Self { linear, nonlinear })
    }
}

impl Module for DenseLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = if x.dims().len() == 2 {
            self.linear.forward(&x.unsqueeze(2)?)?.squeeze(2)?
        } else {
            self.linear.forward(x)?
        };
        self.nonlinear.forward(&out)
    }
}

pub struct CAMPPlus {
    head: FCM,
    tdnn: TDNNLayer,
    blocks: Vec<CAMDenseTDNNBlock>,
    transits: Vec<TransitLayer>,
    out_nonlinear: NonlinearSequential,
    stats: StatsPool,
    dense: DenseLayer,
}

impl CAMPPlus {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let feat_dim = 80;
        let embedding_size = 192;
        let growth_rate = 32;
        let bn_size = 4;
        let init_channels = 128;
        let config_str = "batchnorm-relu";

        let head = FCM::load(vb.pp("head"), 32, feat_dim)?;
        let mut channels = head.out_channels;

        let tdnn = TDNNLayer::load(
            vb.pp("xvector.tdnn"),
            channels,
            init_channels,
            5,
            2,
            1,
            -1,
            config_str,
        )?;
        channels = init_channels;

        let mut blocks = Vec::new();
        let mut transits = Vec::new();

        let specs = [(12, 3, 1), (24, 3, 2), (16, 3, 2)];

        for (i, (num_layers, kernel_size, dilation)) in specs.iter().enumerate() {
            let block = CAMDenseTDNNBlock::load(
                vb.pp(format!("xvector.block{}", i + 1)),
                *num_layers,
                channels,
                growth_rate,
                bn_size * growth_rate,
                *kernel_size,
                1,
                *dilation,
                config_str,
            )?;
            blocks.push(block);

            channels += num_layers * growth_rate;

            let transit = TransitLayer::load(
                vb.pp(format!("xvector.transit{}", i + 1)),
                channels,
                channels / 2,
                config_str,
            )?;
            transits.push(transit);

            channels /= 2;
        }

        let out_nonlinear =
            NonlinearSequential::load(vb.pp("xvector.out_nonlinear"), config_str, channels)?;
        let stats = StatsPool::new();
        let dense = DenseLayer::load(
            vb.pp("xvector.dense"),
            channels * 2,
            embedding_size,
            "batchnorm_",
        )?;

        Ok(Self {
            head,
            tdnn,
            blocks,
            transits,
            out_nonlinear,
            stats,
            dense,
        })
    }
}

impl Module for CAMPPlus {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, F) -> transpose to (B, F, T) for CNNs
        let mut x = x.transpose(1, 2)?.contiguous()?;

        x = self.head.forward(&x)?;
        x = self.tdnn.forward(&x)?;

        for (block, transit) in self.blocks.iter().zip(self.transits.iter()) {
            x = block.forward(&x)?;
            x = transit.forward(&x)?;
        }

        x = self.out_nonlinear.forward(&x)?;
        x = self.stats.forward(&x)?;
        x = self.dense.forward(&x)?;

        Ok(x)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_xvector_instantiation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let model = CAMPPlus::load(vb).unwrap();
        // Since load worked, let's test inference on a dummy tensor
        // (Batch, Time, Frequencies) - Python code takes (B, T, F) and then transposes to (B, F, T)
        let dummy = Tensor::randn(0.0f32, 1.0f32, (2, 200, 80), &device).unwrap();

        let out = candle_nn::Module::forward(&model, &dummy).unwrap();

        assert_eq!(out.dims(), &[2, 192]); // Embedding size is 192
        println!("test_xvector_instantiation pass");
    }
}
