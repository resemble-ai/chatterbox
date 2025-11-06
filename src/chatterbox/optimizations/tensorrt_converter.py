"""
TensorRT Conversion for S3Gen Vocoder

Provides 2-5x speedup for vocoder with FP16/INT8 precision.

Requirements:
    pip install tensorrt torch2trt onnx onnx-simplifier
"""
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class TensorRTVocoder:
    """
    TensorRT-optimized vocoder wrapper

    Converts S3Gen vocoder to TensorRT for maximum speed.
    """
    def __init__(
        self,
        trt_engine_path: str,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        """
        Args:
            trt_engine_path: Path to serialized TensorRT engine
            device: CUDA device
            use_fp16: Use FP16 precision
        """
        self.device = device
        self.use_fp16 = use_fp16

        # Load TensorRT
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "TensorRT not found. Install with: "
                "pip install tensorrt pycuda"
            )

        self.trt = trt
        self.cuda = cuda

        # Load engine
        self.engine = self._load_engine(trt_engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

        logger.info(f"TensorRT vocoder loaded from {trt_engine_path}")

    def _load_engine(self, engine_path: str):
        """Load serialized TensorRT engine"""
        with open(engine_path, 'rb') as f:
            runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """Allocate GPU memory for inputs/outputs"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = self.cuda.Stream()

        for binding in self.engine:
            size = self.trt.volume(self.engine.get_binding_shape(binding))
            dtype = self.trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Run TensorRT inference

        Args:
            mel_spectrogram: (batch, mel_bins, time) tensor

        Returns:
            waveform: (batch, samples) tensor
        """
        # Copy input to GPU
        np_input = mel_spectrogram.cpu().numpy().ravel()
        np.copyto(self.inputs[0]['host'], np_input)

        # Transfer to device
        self.cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Transfer output back
        self.cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )

        # Synchronize
        self.stream.synchronize()

        # Convert to torch tensor
        output = torch.from_numpy(self.outputs[0]['host']).to(self.device)

        # Reshape to (batch, samples)
        batch_size = mel_spectrogram.shape[0]
        output = output.reshape(batch_size, -1)

        return output


def export_vocoder_to_onnx(
    vocoder: nn.Module,
    output_path: str,
    sample_input: torch.Tensor,
    opset_version: int = 17,
    simplify: bool = True,
) -> str:
    """
    Export vocoder to ONNX format

    Args:
        vocoder: PyTorch vocoder model
        output_path: Path to save ONNX model
        sample_input: Sample input for tracing
        opset_version: ONNX opset version
        simplify: Apply onnx-simplifier

    Returns:
        Path to ONNX model
    """
    import onnx
    import onnxruntime

    logger.info(f"Exporting vocoder to ONNX: {output_path}")

    # Export
    vocoder.eval()
    with torch.no_grad():
        torch.onnx.export(
            vocoder,
            sample_input,
            output_path,
            opset_version=opset_version,
            input_names=['mel_spectrogram'],
            output_names=['waveform'],
            dynamic_axes={
                'mel_spectrogram': {0: 'batch', 2: 'time'},
                'waveform': {0: 'batch', 1: 'samples'},
            },
            export_params=True,
            do_constant_folding=True,
        )

    # Simplify
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify

            logger.info("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx_model_simplified, check = onnx_simplify(onnx_model)

            if check:
                onnx.save(onnx_model_simplified, output_path)
                logger.info("ONNX model simplified")
            else:
                logger.warning("ONNX simplification failed")
        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping")

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX model exported and verified: {output_path}")

    return output_path


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    use_fp16: bool = True,
    use_int8: bool = False,
    max_batch_size: int = 8,
    max_workspace_size: int = 1 << 30,  # 1 GB
) -> str:
    """
    Build TensorRT engine from ONNX model

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        use_fp16: Enable FP16 precision
        use_int8: Enable INT8 precision (requires calibration)
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace memory

    Returns:
        Path to TensorRT engine
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT not found. Install from NVIDIA")

    logger.info(f"Building TensorRT engine: {engine_path}")
    logger.info(f"  FP16: {use_fp16}, INT8: {use_int8}")

    # Create builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)

    # Create network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            logger.error("Failed to parse ONNX")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise RuntimeError("ONNX parsing failed")

    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")

    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # TODO: Add INT8 calibrator
        logger.info("INT8 mode enabled (requires calibration)")

    # Build engine
    logger.info("Building TensorRT engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Serialize and save
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    logger.info(f"TensorRT engine saved: {engine_path}")
    return engine_path


def convert_vocoder_to_tensorrt(
    vocoder: nn.Module,
    output_dir: str,
    sample_input: torch.Tensor,
    use_fp16: bool = True,
    use_int8: bool = False,
) -> TensorRTVocoder:
    """
    Complete pipeline: PyTorch -> ONNX -> TensorRT

    Args:
        vocoder: PyTorch vocoder model
        output_dir: Directory to save converted models
        sample_input: Sample input for tracing
        use_fp16: Enable FP16
        use_int8: Enable INT8

    Returns:
        TensorRTVocoder instance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = str(output_dir / "vocoder.onnx")
    engine_path = str(output_dir / "vocoder.trt")

    # Step 1: Export to ONNX
    logger.info("Step 1/2: Exporting to ONNX...")
    export_vocoder_to_onnx(vocoder, onnx_path, sample_input)

    # Step 2: Build TensorRT engine
    logger.info("Step 2/2: Building TensorRT engine...")
    build_tensorrt_engine(
        onnx_path,
        engine_path,
        use_fp16=use_fp16,
        use_int8=use_int8,
    )

    # Load and return
    logger.info("Loading TensorRT vocoder...")
    return TensorRTVocoder(engine_path, use_fp16=use_fp16)


# INT8 Calibrator for TensorRT
class VocoderCalibrator:
    """
    INT8 calibration for TensorRT

    Collects statistics from representative data.
    """
    def __init__(
        self,
        calibration_data: list,
        cache_file: str = "vocoder_calibration.cache",
    ):
        """
        Args:
            calibration_data: List of sample mel spectrograms
            cache_file: Path to save calibration cache
        """
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0

        # Allocate buffer
        self.device_input = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Return a batch of data for calibration"""
        if self.current_index >= len(self.calibration_data):
            return None

        batch = self.calibration_data[self.current_index]
        self.current_index += 1

        # Copy to GPU
        if self.device_input is None:
            import pycuda.driver as cuda
            self.device_input = cuda.mem_alloc(batch.nbytes)

        import pycuda.driver as cuda
        cuda.memcpy_htod(self.device_input, batch)

        return [int(self.device_input)]

    def read_calibration_cache(self):
        """Read calibration cache if it exists"""
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration cache"""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


if __name__ == "__main__":
    # Example usage
    logger.info("TensorRT Vocoder Converter")
    logger.info("=" * 60)

    # This would be called with actual vocoder model
    print("""
    Usage example:

    from chatterbox.models.s3gen import S3Gen
    from chatterbox.optimizations.tensorrt_converter import convert_vocoder_to_tensorrt

    # Load vocoder
    vocoder = S3Gen()
    vocoder.load_state_dict(...)
    vocoder.eval()

    # Sample input (batch, mel_bins, time)
    sample_input = torch.randn(1, 80, 100).cuda()

    # Convert
    trt_vocoder = convert_vocoder_to_tensorrt(
        vocoder.mel2wav,
        output_dir="./trt_models",
        sample_input=sample_input,
        use_fp16=True,
    )

    # Use
    mel = torch.randn(1, 80, 200).cuda()
    wav = trt_vocoder(mel)
    """)
