# Context Managers in Python

Context managers allow for clean and reliable acquisition and release of resources in Python.
This section demonstrates the use and implementation of context managers using the `with` statement.

## ✅ Topics Covered

- What is a context manager
- Using `with` for resource management
- Implementing context managers using a class
- Exception handling in context managers
- Implementing context managers using generator functions

## Files

| File                              | Description                                              |
| --------------------------------- | -------------------------------------------------------- |
| `file_with_statement.py`          | Basic usage of `with open(...)`                          |
| `managed_file_class.py`           | Custom context manager using a class                     |
| `managed_file_class_exception.py` | Handle exceptions in class-based context manager         |
| `managed_file_class_handled.py`   | Suppress exceptions by returning True in `__exit__`      |
| `managed_file_generator.py`       | Custom context manager using `@contextmanager` decorator |

## References

- [Python docs: contextlib](https://docs.python.org/3/library/contextlib.html)
