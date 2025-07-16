import numpy as np
import gradio as gr


def randomize_seed(seed, randomize_seed):
    if randomize_seed:
        return np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    else:
        return int(seed)


def randomize_seed_ui():
    with gr.Accordion("Seed", open=True), gr.Group(), gr.Row(equal_height=True):
        seed_input = gr.Textbox(scale=9, container=False, show_label=False, show_copy_button=True, value="-1")
        randomize_seed_checkbox = gr.Checkbox(scale=1, label="Randomize seed", value=True)

    return (
        seed_input,
        {
            "fn": randomize_seed,
            "inputs": [seed_input, randomize_seed_checkbox],
            "outputs": [seed_input],
        },
    )
