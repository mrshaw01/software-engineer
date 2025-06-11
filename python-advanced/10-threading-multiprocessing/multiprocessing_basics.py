"""
Demonstrates foundational multiprocessing concepts in Python:
- Process creation and joining
- Shared memory with Value and Array
- Preventing race conditions with Locks
- Process-safe Queues
- Using a Pool for parallel mapping
"""

from multiprocessing import Array
from multiprocessing import Lock
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Value
import os
import time


def add_100(number, lock):
    for _ in range(100):
        time.sleep(0.01)
        with lock:
            number.value += 1


def add_100_array(numbers, lock):
    for _ in range(100):
        time.sleep(0.01)
        for i in range(len(numbers)):
            with lock:
                numbers[i] += 1


def square_worker(numbers, queue):
    for i in numbers:
        queue.put(i * i)


def negative_worker(numbers, queue):
    for i in numbers:
        queue.put(-i)


def cube(x):
    return x * x * x


def basic_process_run():
    print("\n--- Basic multiprocessing with Process and Lock ---")
    shared_number = Value("i", 0)
    shared_array = Array("d", [0.0, 100.0, 200.0])
    lock = Lock()

    print(f"Initial shared number: {shared_number.value}")
    print(f"Initial shared array: {list(shared_array)}")

    p1 = Process(target=add_100, args=(shared_number, lock))
    p2 = Process(target=add_100, args=(shared_number, lock))
    p3 = Process(target=add_100_array, args=(shared_array, lock))
    p4 = Process(target=add_100_array, args=(shared_array, lock))

    for p in [p1, p2, p3, p4]:
        p.start()
    for p in [p1, p2, p3, p4]:
        p.join()

    print(f"Final shared number: {shared_number.value}")
    print(f"Final shared array: {list(shared_array)}")


def queue_communication():
    print("\n--- Multiprocessing Queue Communication ---")
    numbers = range(1, 6)
    queue = Queue()

    p1 = Process(target=square_worker, args=(numbers, queue))
    p2 = Process(target=negative_worker, args=(numbers, queue))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    while not queue.empty():
        print(queue.get())


def pool_example():
    print("\n--- Using multiprocessing.Pool for parallel map ---")
    with Pool() as pool:
        numbers = list(range(10))
        results = pool.map(cube, numbers)
        print(results)


def square_numbers():
    for i in range(1000):
        result = i * i


def basic_process_spawn():
    print("\n--- Spawn basic CPU-bound processes ---")
    processes = []
    num_processes = os.cpu_count()

    for _ in range(num_processes):
        process = Process(target=square_numbers)
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    basic_process_spawn()
    basic_process_run()
    queue_communication()
    pool_example()
    print("\nAll multiprocessing examples completed.")
