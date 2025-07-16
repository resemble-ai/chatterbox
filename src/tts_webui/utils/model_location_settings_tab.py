import gradio as gr
import os


def model_location_settings():
    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("## Model Location Settings (Experimental!)")

            gr.Markdown("- Requires restart to apply")

            gr.Markdown(
                """
    1. **HUGGINGFACE_HUB_CACHE**: 
    - This environment variable is used to specify the location of the Hugging Face cache, which stores downloaded models and other assets used by Hugging Face Transformers library.
    - By default, the cache directory is usually set to your user's home directory.
    - You can customize the cache directory by either providing the `cache_dir` argument in the methods of Hugging Face Transformers library, or by setting this environment variable to the desired path.

    2. **HF_HOME**: 
    - This environment variable is also related to the Hugging Face cache.
    - It allows you to set a custom directory for the Hugging Face cache, overriding the default location (usually the user's home directory).
    - If `HUGGINGFACE_HUB_CACHE` is not set, this environment variable can be used as an alternative to specify the Hugging Face cache location.

    3. **TORCH_HOME**:
    - This environment variable allows you to set a custom directory for the Torch Hub cache, similar to `PATH_TO_HUB_DIR`.
    - If `PATH_TO_HUB_DIR` is not set, you can use this environment variable to specify the Torch Hub cache location.

    4. **XDG_CACHE_HOME**:
    - This environment variable allows you to set a custom directory for caching various applications, including the Torch Hub cache.
    - If `PATH_TO_HUB_DIR` and `TORCH_HOME` are not set, you can use this environment variable to specify the Torch Hub cache location.

    Please note that the "tts-webui Directory" is currently fixed and cannot be changed. This directory contains the TTS (Text-to-Speech) models used by the web UI. The models are stored inside the `data/models/` directory within the `tts-webui` directory.

    Using these environment variables allows for better management and sharing of cached assets between different installations and projects, making it easier to access and reuse models and data across different user interfaces and projects.
    """
            )
        with gr.Column():
            # Hugging Face cache
            gr.Markdown("### Hugging Face Cache")
            model_location_hf_env_var = gr.Textbox(
                label="Environment: HUGGINGFACE_HUB_CACHE",
                value=os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
                placeholder="Unset",
            )
            model_location_hf_env_var2 = gr.Textbox(
                label="Environment: HF_HOME",
                value=os.environ.get("HF_HOME", ""),
                placeholder="Unset",
            )

            # Torch Hub cache
            gr.Markdown("### Torch Hub Cache")
            # model_location_th_set_dir = gr.Textbox(
            #     label="Environment: PATH_TO_HUB_DIR", value="Default"
            # )  # for hub.set_dir(<PATH_TO_HUB_DIR>)
            model_location_th_home = gr.Textbox(
                label="Environment: TORCH_HOME",
                value=os.environ.get("TORCH_HOME", ""),
                placeholder="Unset, default: ~/.cache/torch/",
            )
            model_location_th_xdg = gr.Textbox(
                label="Environment: XDG_CACHE_HOME",
                value=os.environ.get("XDG_CACHE_HOME", ""),
                placeholder="Unset, default: ~/.cache/",
            )
            # tts-webui directory
            gr.Markdown("### tts-webui Directory (can't be changed yet)")
            gr.Textbox(
                label="Model Location (TTS Default)",
                value="./tts-webui/data/models/",
                interactive=False,
            )

            inputs = [
                model_location_hf_env_var,
                model_location_hf_env_var2,
                model_location_th_home,
                model_location_th_xdg,
            ]

            save_beacon = gr.Markdown()

            def save_environment_variables2(
                model_location_hf_env_var,
                model_location_hf_env_var2,
                model_location_th_home,
                model_location_th_xdg,
            ):
                os.environ["HUGGINGFACE_HUB_CACHE"] = str(model_location_hf_env_var)
                os.environ["HF_HOME"] = str(model_location_hf_env_var2)
                os.environ["TORCH_HOME"] = str(model_location_th_home)
                os.environ["XDG_CACHE_HOME"] = str(model_location_th_xdg)
                from tts_webui.utils.setup_or_recover import generate_env, write_env

                write_env(
                    generate_env(
                        model_location_hf_env_var=model_location_hf_env_var,
                        model_location_hf_env_var2=model_location_hf_env_var2,
                        model_location_th_home=model_location_th_home,
                        model_location_th_xdg=model_location_th_xdg,
                    )
                )
                # os._exit(0)
                return "saved"

            for i in inputs:
                i.change(
                    fn=save_environment_variables2, inputs=inputs, outputs=[save_beacon]
                )


def model_location_settings_tab():
    with gr.Tab("Model Location Settings"):
        model_location_settings()
