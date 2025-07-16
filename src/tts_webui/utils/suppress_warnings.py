import logging
import warnings


def suppress_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Using the update method is deprecated. Simply return a new object instead",
    )
    warnings.filterwarnings(
        "ignore",
        message="Trying to convert audio automatically from float32 to 16-bit int format.",
    )
    warnings.filterwarnings(
        "ignore",
        message="Trying to convert audio automatically from int32 to 16-bit int format.",
    )
    # UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    warnings.filterwarnings(
        "ignore",
        message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.",
    )
    # FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
    warnings.filterwarnings(
        "ignore",
        message="transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations",
    )
    # UserWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
    # warnings.warn("torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")
    warnings.filterwarnings(
        "ignore",
        message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
    )

    # suppress warning from logging "A matching Triton is not available, some optimizations will not be enabled"
    # suppress warning from logging "Triton is not available, some optimizations will not be enabled."
    logging.getLogger("xformers").addFilter(
        lambda record: "Triton is not available" not in record.getMessage()
    )
