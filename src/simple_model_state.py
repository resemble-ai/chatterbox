import functools
from typing import Dict, Any, Optional


class SimpleModelState:
    def __init__(self):
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None

    def set_model(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name

    def get_model(self) -> Optional[Any]:
        return self.model

    def is_model_loaded(self, model_name: str) -> bool:
        return self.model is not None and self.model_name == model_name

    def clear(self):
        self.model = None
        self.model_name = None


# Global model cache
_model_cache: Dict[str, SimpleModelState] = {}


def simple_manage_model_state(model_namespace: str):
    """Simplified decorator to manage model state without external dependencies."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(model_name, *args, **kwargs):
            global _model_cache

            if model_namespace not in _model_cache:
                _model_cache[model_namespace] = SimpleModelState()

            model_state = _model_cache[model_namespace]

            if not model_state.is_model_loaded(model_name):
                print(f"Loading model '{model_name}' in namespace '{model_namespace}'...")
                # Clear any existing model
                model_state.clear()
                # Load new model
                model = func(model_name, *args, **kwargs)
                model_state.set_model(model, model_name)
            else:
                print(f"Using cached model '{model_name}' in namespace '{model_namespace}'.")

            return model_state.get_model()

        return wrapper

    return decorator