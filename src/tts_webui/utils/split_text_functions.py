#def split_by_lines(prompt: str):
#    prompts = prompt.split("\n")
#    prompts = [p.strip() for p in prompts]
#    prompts = [p for p in prompts if len(p) > 0]
#    return prompts


def split_by_length_simple(prompt: str):
    return [prompt[i : i + 200] for i in range(0, len(prompt), 200)]


try:
    from tortoise.utils.text import split_and_recombine_text
except ImportError:

    def split_and_recombine_text(
        text: str, desired_length: int = 200, max_length: int = 300
    ):
        return split_by_length_simple(text)

    # split_and_recombine_text = split_by_length_simple


import re
def split_by_lines(prompt: str):
    #prompts = re.split(r"(?<=[.!?])\s+", prompt)
    prompts = re.split(r'(?<=[.?!])\s*(?![.\w"\'\d])', prompt)
    # Filter out empty strings that might result from the split
    prompts = [p.strip() for p in prompts if p.strip()]
    #prompts = [p.strip() for p in prompts]
    #prompts = [p for p in prompts if len(p) > 0]
    return prompts