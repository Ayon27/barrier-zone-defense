"""Write to both stdout and a file simultaneously."""

import os
import sys


class Tee:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8", buffering=1)  # Line-buffered
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self.stdout
        self.file.close()


def run_with_logging(main_fn, results_path):
    with Tee(results_path):
        main_fn()
    print(f"Results saved to: {results_path}")
