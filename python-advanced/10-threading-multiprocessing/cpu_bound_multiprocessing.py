"""Demonstrates multiprocessing on a CPU-bound task.
This approach achieves parallel execution and bypasses the GIL.
"""

from multiprocessing import cpu_count
from multiprocessing import Process


def compute_squares():
    for _ in range(1_000_000):
        _ = 42**2


if __name__ == "__main__":
    processes = []
    for _ in range(cpu_count()):
        process = Process(target=compute_squares)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
