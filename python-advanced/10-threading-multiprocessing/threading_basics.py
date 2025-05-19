"""
Demonstrates basic and intermediate threading patterns in Python, including:
- Creating and managing multiple threads
- Preventing race conditions with Lock
- Daemon threads and their behavior
- Using Queue for thread-safe communication
"""

import time
from queue import Queue
from threading import Lock, Thread, current_thread

# Shared variable across threads
database_value = 0


def unsafe_increase():
    """A thread-unsafe function that reads, modifies, and writes a global variable."""
    global database_value
    local_copy = database_value
    local_copy += 1
    time.sleep(0.1)
    database_value = local_copy


def safe_increase(lock):
    """Thread-safe version using Lock as a context manager."""
    global database_value
    with lock:
        local_copy = database_value
        local_copy += 1
        time.sleep(0.1)
        database_value = local_copy


def lock_example():
    """Example using threads with Lock to prevent race conditions."""
    global database_value
    database_value = 0
    lock = Lock()

    t1 = Thread(target=safe_increase, args=(lock,))
    t2 = Thread(target=safe_increase, args=(lock,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("[Lock Example] Final database_value:", database_value)


def queue_worker(q: Queue, lock: Lock):
    """Worker function that processes tasks from a queue."""
    while True:
        value = q.get()
        with lock:
            print(f"[{current_thread().name}] processed value: {value}")
        q.task_done()


def queue_example():
    """Example using Queue with multiple daemon threads."""
    q = Queue()
    lock = Lock()
    num_threads = 5

    for i in range(num_threads):
        t = Thread(target=queue_worker, args=(q, lock), name=f"Worker-{i+1}", daemon=True)
        t.start()

    for i in range(10):
        q.put(i)

    q.join()
    print("[Queue Example] All tasks processed.")


def basic_thread_example():
    """Start multiple threads for a simple CPU-bound function."""

    def square_numbers():
        for i in range(10000):
            i * i

    threads = []
    for _ in range(5):
        t = Thread(target=square_numbers)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("[Basic Example] Completed all square computations.")


if __name__ == "__main__":
    print("--- Basic Threading Example ---")
    basic_thread_example()

    print("\n--- Threading with Lock Example ---")
    lock_example()

    print("\n--- Threading with Queue Example ---")
    queue_example()
    print("Main thread done.")
