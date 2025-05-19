"""Demonstrates shared memory and the limitations of the GIL in multi-threaded Python.
All threads increment the same counter; expected result is not guaranteed due to race conditions.
"""

import dis
import threading

counter = 0


def f():
    global counter
    counter += 1


def increment():
    global counter
    for _ in range(1_000_000):
        counter += 1


if __name__ == "__main__":
    dis.dis(f)
    threads = []
    for _ in range(2):
        thread = threading.Thread(target=increment)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Expected counter = 2_000_000")
    print("Actual counter  =", counter)
