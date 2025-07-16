import gradio as gr
import torch


def gpu_info_tab():
    with gr.Tab("GPU Info") as gpu_info_tab:
        gpu_info = gr.Markdown("")

        gr.Button("Refresh").click(
            fn=refresh_gpu_info,
            outputs=gpu_info,
            api_name="refresh_gpu_info",  # , every=1
        )
        gpu_info_tab.select(
            fn=refresh_gpu_info,
            outputs=gpu_info,
        )

        gr.Button("API_GET_GPU_INFO", visible=False).click(
            fn=get_gpu_info,
            outputs=[gr.JSON(None, visible=False)],
            api_name="get_gpu_info",
        )


def get_gpu_info():
    if not torch.cuda.is_available():
        return []
    device_count = torch.cuda.device_count()
    return [get_gpu_info_idx(idx) for idx in range(device_count)]


def get_pynvml_fields(idx=0):
    # check if pynvml is installed
    try:
        # import pynvml
        return {
            "temperature": torch.cuda.temperature(idx),
            "power_draw": torch.cuda.power_draw(idx) / 1000,
            "utilization": torch.cuda.utilization(idx),
        }
    # except ImportError:
    except:
        return {
            "temperature": 0,
            "power_draw": 0,
            "utilization": 0,
        }


def get_gpu_info_idx(idx=0):
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "vram": torch.cuda.get_device_properties(idx).total_memory / 1024**2,
        "name": torch.cuda.get_device_properties(idx).name,
        "cuda_capabilities": torch.cuda.get_device_capability(idx),
        "used_vram": torch.cuda.memory_allocated(idx) / 1024**2,
        "used_vram_total": (
            torch.cuda.mem_get_info(idx)[1] - torch.cuda.mem_get_info(idx)[0]
        )
        / 1024**2,
        "cached_vram": torch.cuda.memory_reserved(idx) / 1024**2,
        "idx": idx,
        "multi_processor_count": torch.cuda.get_device_properties(
            idx
        ).multi_processor_count,
        **get_pynvml_fields(idx),
    }


def render_gpu_info(gpu_info):
    if isinstance(gpu_info, dict):
        return f"""VRAM: {gpu_info['vram']} MB

Used VRAM: {gpu_info['used_vram']} MB

Total Used VRAM: {gpu_info['used_vram_total']} MB

Name: {gpu_info['name']}

CUDA Capabilities: {gpu_info['cuda_capabilities']}

Cached VRAM: {gpu_info['cached_vram']} MB

Torch Version: {gpu_info['torch_version']}"""
    else:
        return gpu_info


def refresh_gpu_info():
    return "".join([render_gpu_info(x) for x in get_gpu_info()])


if __name__ == "__main__":
    if "demo" in locals():
        demo.close()  # type: ignore
    with gr.Blocks() as demo:
        gpu_info_tab()

    demo.launch(
        server_port=7770,
    )
