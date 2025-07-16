import datetime


def get_date_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")
