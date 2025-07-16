import time


class LoadingIndicator:
    def __init__(self, title_name, skipped=False):
        self.title_name = title_name
        self.skipped = skipped

    def __enter__(self):
        print(f"Loading {self.title_name.ljust(35, '.')}...", end="")
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.skipped:
            print(f"{' ' * 6} skipped.")
            return
        elapsed_time = time.time() - self.start_time
        seconds = f"{elapsed_time:.2f}"
        seconds = seconds.replace("0.", " .")
        print(f"{seconds.rjust(6, ' ')} seconds.")
