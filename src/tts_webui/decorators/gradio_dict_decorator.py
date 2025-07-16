import inspect
import functools


def gradio_dict_decorator(fn, gradio_fn_input_dictionary, outputs):
    """
    A helper function that allows specifying gradio inputs and outputs as a dictionary.

    This function supports both regular functions and generator functions (using yield).

    For example:
    text = gr.Textbox(label="Text")
    description = gr.Textbox(label="Context")

    audio_out = gr.Audio(label="Parler-TTS generation", type="numpy", elem_id="audio_out")

    gradion_fn_input_dictionary = {
        text: "text",
        description: "description",
    }

    outputs = { "audio_out": audio_out }

    def fn(text, description):
        yield { "audio_out": (SAMPLE_RATE, text + description) }
        return { "audio_out": (SAMPLE_RATE, text + description) }

    gr.Button("Generate").click(
        inputs={*gradion_fn_input_dictionary},
        outputs=[v for _, v in outputs.items()],
        fn=gradio_dict_decorator(fn, gradion_fn_input_dictionary, outputs)
    )

    """

    def unmap_outputs(result_dict):
        return {v: result_dict[k] for k, v in outputs.items()}

    if inspect.isgeneratorfunction(fn):

        def wrapper(dict_args):
            mapped_args = {
                gradio_fn_input_dictionary[k]: v for k, v in dict_args.items()
            }
            for result_dict in fn(**mapped_args, outputs=outputs):
                yield unmap_outputs(result_dict)

        return wrapper
    else:

        def wrapper_b(dict_args):
            mapped_args = {
                gradio_fn_input_dictionary[k]: v for k, v in dict_args.items()
            }
            result_dict = fn(**mapped_args, outputs=outputs)
            return unmap_outputs(result_dict)

        return wrapper_b


def _apply_sig(fn, input_dict):
    """
    Assigns a signature to a function.

    For example:

    input_dict = {
        "text": "text",
        "description": "description",
    }

    def fn(**kwargs):
        return kwargs["text"] + kwargs["description"]

    fn = _apply_sig(fn, input_dict)

    def fn2(text, description):
        return text + description

    fn == fn2
    """
    # Assign the new signature to the wrapper function
    fn.__signature__ = inspect.Signature(
        [
            # Create a new signature based on input_dict
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in input_dict.values()
        ]
    )
    return fn


def _get_mapped_args(inputs: dict, list_args):
    """
    Takes a list of arguments and maps them to a dictionary.

    For example:
    inputs = {
        text: "text",
        description: "description",
    }

    list_args = ["hello", "world"]

    _get_mapped_args(list_args) == {
        "text": "hello",
        "description": "world",
    }
    """
    return {k: v for k, v in zip(list(inputs.values()), list_args)}


def dictionarize(fn, inputs: dict, outputs: dict, **kwargs):
    def unmap_outputs(result_dict):
        return {v: result_dict[k] for k, v in outputs.items()}

    def get_wrapper():
        if inspect.isgeneratorfunction(fn):

            def gen_wrapper(*list_args):
                mapped_args = _get_mapped_args(inputs, list_args)
                print(mapped_args)
                for result_dict in fn(
                    **_get_mapped_args(inputs, list_args), outputs=outputs
                ):
                    yield unmap_outputs(result_dict)

            return _apply_sig(gen_wrapper, inputs)
        else:

            def wrapper(*list_args):
                mapped_args = _get_mapped_args(inputs, list_args)
                print(mapped_args)
                result_dict = fn(**_get_mapped_args(inputs, list_args), outputs=outputs)
                return unmap_outputs(result_dict)

            return _apply_sig(wrapper, inputs)

    return {
        "fn": get_wrapper(),
        "inputs": list(inputs.keys()),
        "outputs": list(outputs.values()),
        **kwargs,
    }


def _apply_sig_with_defaults(fn, input_dict, original_fn=None):
    """
    Assigns a signature to a function, preserving default values from the original signature.
    """
    # Get the original signature to extract default values
    # If original_fn is provided, use that; otherwise inspect fn itself
    source_fn = original_fn if original_fn else fn
    original_sig = inspect.signature(source_fn)
    original_params = original_sig.parameters

    # Create new parameters with preserved defaults
    new_params = []
    for name in input_dict.values():
        if name in original_params:
            # Use the original parameter with its default value
            original_param = original_params[name]
            # Only set default if it's not Parameter.empty
            default = (
                original_param.default
                if original_param.default != inspect.Parameter.empty
                else inspect.Parameter.empty
            )
            new_param = inspect.Parameter(
                name=name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=original_param.annotation,
            )
        else:
            # Create new parameter without default
            new_param = inspect.Parameter(
                name=name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        new_params.append(new_param)

    # Assign the new signature to the wrapper function
    fn.__signature__ = inspect.Signature(new_params)
    return fn


def dictionarize_wraps(fn, inputs: dict, outputs: dict, **kwargs):
    def unmap_outputs(result_dict):
        return {v: result_dict[k] for k, v in outputs.items()}

    def get_wrapper():
        if inspect.isgeneratorfunction(fn):

            def gen_wrapper(*list_args):
                mapped_args = _get_mapped_args(inputs, list_args)
                print(mapped_args)
                for result_dict in fn(
                    **_get_mapped_args(inputs, list_args), outputs=outputs
                ):
                    yield unmap_outputs(result_dict)

            return _apply_sig_with_defaults(gen_wrapper, inputs, original_fn=fn)
        else:

            # @functools.wraps(fn)
            def wrapper(*list_args):
                mapped_args = _get_mapped_args(inputs, list_args)
                print(mapped_args)
                result_dict = fn(**_get_mapped_args(inputs, list_args), outputs=outputs)
                return unmap_outputs(result_dict)

            return _apply_sig_with_defaults(wrapper, inputs, original_fn=fn)

    return {
        "fn": get_wrapper(),
        "inputs": list(inputs.keys()),
        "outputs": list(outputs.values()),
        **kwargs,
    }
