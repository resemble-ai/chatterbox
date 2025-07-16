import os
import re


def generate_pretty_name(directory: str):
    name = directory
    # remove directory name
    name = os.path.basename(name)
    # remove the date and time
    date_regex = r"([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})"
    name = re.sub(date_regex, "", name)
    # remove "audio"
    name = name.replace("audio", "")
    # replace _long with (long)
    name = name.replace("_long", "(long)")
    # replace continued_generation with history: continued
    name = name.replace("continued_generation", "history: continued")
    # replace the new continued_from_ with history:
    name = name.replace("from_", "history: ")
    # replace none with history: none
    name = name.replace("None", "history: none")
    # __bark__None
    name = name.replace("__", " ")
    # bark None
    return name.strip().capitalize()


if __name__ == "__main__":
    # outputs\2023-06-02_10-54-24__bark__None
    dirname = "outputs\\2023-06-02_10-54-24__bark__None"
    print(generate_pretty_name(dirname))

    # outputs\audio__bark__continued_generation__2023-05-21_15-36-14
    dirname = "outputs\\audio__bark__continued_generation__2023-05-21_15-36-14"
    print(generate_pretty_name(dirname))

    # outputs\audio__bark__continued_generation__2023-05-21_15-36-14_long
    dirname = "outputs\\audio__bark__continued_generation__2023-05-21_15-36-14_long"
    print(generate_pretty_name(dirname))
