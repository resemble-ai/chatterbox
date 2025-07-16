def prompt_to_title(prompt):
    return (
        prompt.replace(" ", "_")
        .replace(":", "_")
        .replace("'", "_")
        .replace('"', "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(",", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("?", "_")
        .replace("!", "_")
        .replace("/", "_")
        .replace("\n", "_")
        # only first 15 characters
        .replace("__", "_")[:15]
    )
