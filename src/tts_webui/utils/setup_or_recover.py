import os


def env_entry(name, value, comment, null_if_empty=True):
    return f"# {comment}\n{'# ' if null_if_empty and not value else ''}{name}={value}\n"

def generate_env(
    *,
    model_location_hf_env_var: str = "",
    model_location_hf_env_var2: str = "",
    model_location_th_home: str = "",
    model_location_th_xdg: str = "",
    # data\models\rvc\checkpoints\Alina_Gray-20230627T032329Z-001\Alina_Gray.pth
    rvc_weight_root: str = "data/models/rvc/checkpoints",
    rvc_weight_uvr5_root: str = "data/models/rvc/uvr5_weights",
    rvc_index_root: str = "data/models/rvc/checkpoints",
    rvc_outside_index_root: str = "data/models/rvc/checkpoints",
    rvc_rmvpe_root: str = "data/models/rvc/rmvpe",
):
    model_location_hf_env_var = model_location_hf_env_var or os.environ.get(
        "HUGGINGFACE_HUB_CACHE", ""
    )
    model_location_hf_env_var2 = model_location_hf_env_var2 or os.environ.get(
        "HF_HOME", ""
    )
    model_location_th_home = model_location_th_home or os.environ.get("TORCH_HOME", "")
    model_location_th_xdg = model_location_th_xdg or os.environ.get(
        "XDG_CACHE_HOME", ""
    )

    rvc_weight_root = rvc_weight_root or os.environ.get("weight_root", "")
    rvc_weight_uvr5_root = rvc_weight_uvr5_root or os.environ.get(
        "weight_uvr5_root", ""
    )
    rvc_index_root = rvc_index_root or os.environ.get("index_root", "")
    rvc_outside_index_root = rvc_outside_index_root or os.environ.get(
        "outside_index_root", ""
    )
    rvc_rmvpe_root = rvc_rmvpe_root or os.environ.get("rmvpe_root", "")

    env = "# This file gets updated automatically from the UI\n\n"
    env += "# If you wish to manually specify any ENV variables, please do so in the .env.user file\n"
    env += "# The variables in .env.user will take PRIORITY!\n\n"

    env += env_entry("USE_TF", "0", "Disable Tensorflow for transformers")

    env += env_entry(
        "HUGGINGFACE_HUB_CACHE",
        model_location_hf_env_var,
        "Environment variable for HuggingFace model location",
    )
    env += env_entry(
        "HF_HOME",
        model_location_hf_env_var2,
        "Environment variable for HuggingFace model location (alternative)",
    )
    env += env_entry(
        "TORCH_HOME",
        model_location_th_home,
        "Default location for Torch Hub models",
    )
    env += env_entry(
        "XDG_CACHE_HOME",
        model_location_th_xdg,
        "Default location for Torch Hub models (alternative)",
    )

    env += "\n"

    env += env_entry(
        "weight_root",
        rvc_weight_root,
        "Root directory for RVC model weights",
    )
    env += env_entry(
        "weight_uvr5_root",
        rvc_weight_uvr5_root,
        "Root directory for RVC model weights (UVR5)",
    )
    env += env_entry(
        "index_root",
        rvc_index_root,
        "Root directory for RVC model indices",
    )
    env += env_entry(
        "outside_index_root",
        rvc_outside_index_root,
        "Root directory for RVC model indices (outside)",
    )
    env += env_entry(
        "rmvpe_root",
        rvc_rmvpe_root,
        "Root directory for RVC model RMVPE",
    )

    env += "\n"

    # CUDA_VISIBLE_DEVICES
    # env += env_entry(
    #     "CUDA_VISIBLE_DEVICES",
    #     "0",
    #     "CUDA device to use (0,1,2,3, etc.)",
    #     null=False,
    # )

    return env


def write_env(text: str):
    with open(".env", "w") as outfile:
        outfile.write(text)


def setup_or_recover():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("favorites"):
        os.makedirs("favorites")
    if not os.path.exists(".env"):
        print("Env file not found. Creating default env.")
        write_env(generate_env())
