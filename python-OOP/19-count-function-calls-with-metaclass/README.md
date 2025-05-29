# 19. Count Function Calls with a Metaclass

This tutorial demonstrates how to use a **Python metaclass** to automatically decorate class methods so that the number of times each method is called is counted.

## What You'll Learn

- What a metaclass is and how it works.
- How to use metaclasses to modify class behavior.
- How to apply a decorator to all methods in a class.
- A simple profiling use case: method call counting.

## Files

| File                        | Description                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| `call_counter_decorator.py` | A standalone version of the call counter decorator.                       |
| `metaclass_counter.py`      | Defines a metaclass that decorates class methods to count function calls. |
| `test_metaclass_counter.py` | Example usage and test of the metaclass-based function call counter.      |

## Run

```bash
python test_metaclass_counter.py
```
