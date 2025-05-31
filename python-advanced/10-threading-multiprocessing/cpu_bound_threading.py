"""Demonstrates threading on a CPU-bound task.
Threading is ineffective here due to the Global Interpreter Lock (GIL).
"""

from threading import Thread


def compute_squares():
    for _ in range(100_000_000):
        _ = 42**2
    print("Finished computing squares.")


if __name__ == "__main__":
    threads = []
    for _ in range(4):
        thread = Thread(target=compute_squares)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
