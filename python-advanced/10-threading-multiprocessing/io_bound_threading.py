"""Demonstrates threading used for I/O-bound tasks.
Threads are ideal for overlapping wait times, e.g., network or file I/O.
"""

import time
from threading import Thread


def simulated_io_task(name):
    print(f"{name} started I/O operation")
    time.sleep(2)
    print(f"{name} completed I/O operation")


if __name__ == "__main__":
    threads = []
    for i in range(5):
        thread = Thread(target=simulated_io_task, args=(f"Thread-{i}",))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
