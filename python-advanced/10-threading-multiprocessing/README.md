# Threading vs Multiprocessing in Python

This module demonstrates the practical differences between **threading** and **multiprocessing** in Python — two powerful tools for concurrency and parallelism. It explains:

- When to use threads (I/O-bound tasks)
- When to use processes (CPU-bound tasks)
- Python’s Global Interpreter Lock (GIL) and its impact
- Practical memory and performance considerations

## Key Differences

| Aspect            | Threading                           | Multiprocessing                         |
| ----------------- | ----------------------------------- | --------------------------------------- |
| Execution Unit    | Thread (lightweight)                | Process (independent)                   |
| Memory Sharing    | Shared memory space                 | Separate memory space                   |
| GIL Effect        | Shared GIL → limited parallelism    | One GIL per process → real parallelism  |
| Start-up Time     | Fast                                | Slower                                  |
| Best for          | I/O-bound tasks (e.g., networking)  | CPU-bound tasks (e.g., computation)     |
| Failure Isolation | Threads can crash the whole process | Process isolation; safer error handling |

## Files

| File                               | Description                                        |
| ---------------------------------- | -------------------------------------------------- |
| `threading_basics.py`              | Thread creation, join, race condition, queue usage |
| `cpu_bound_threading.py`           | Shows threading bottleneck with CPU-bound tasks    |
| `cpu_bound_multiprocessing.py`     | Achieves true parallelism for CPU-bound tasks      |
| `io_bound_threading.py`            | Ideal use of threading for I/O-heavy workloads     |
| `shared_memory_gil_limitations.py` | Demonstrates shared state and GIL effects          |
| `multiprocessing_basics.py`        | Process usage, shared memory, Queue, Pool examples |

## Run Examples

```bash
python threading_basics.py
python multiprocessing_basics.py
python cpu_bound_threading.py
python cpu_bound_multiprocessing.py
```

## Recommended Reading

- [Python `threading`](https://docs.python.org/3/library/threading.html)
- [Python `multiprocessing`](https://docs.python.org/3/library/multiprocessing.html)
- [Global Interpreter Lock (GIL)](https://wiki.python.org/moin/GlobalInterpreterLock)
