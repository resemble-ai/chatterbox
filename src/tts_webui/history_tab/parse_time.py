import datetime
import re


# 2023-05-16_11-45-00
def parse_time(text: str):
    try:
        # extract date and time
        date, time = text.split("_")
        # extract year, month, day
        year, month, day = date.split("-")
        # extract hour, minute, second
        hour, minute, second = time.split("-")
        # create a native date
        return datetime.datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second)
        )
    except Exception:
        print("Error parsing time")
        print(text)
        return None


# audio__bark__None__2023-05-16_11-45-00_long.wav
# audio__tortoise__random__2023-05-31_14-19-13__n0.wav
# Matches the time string in the filename and returns it
def extract_time(filename: str):
    # only match the time string
    regex = r"([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})"
    matches = re.finditer(regex, filename, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        return match.group(1)


def extract_and_parse_time(filename: str):
    return parse_time(extract_time(filename))
