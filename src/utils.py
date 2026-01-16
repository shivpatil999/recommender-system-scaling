import os
import time
import psutil
from contextlib import contextmanager

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

@contextmanager
def timer():
    start = time.time()
    yield lambda: time.time() - start
