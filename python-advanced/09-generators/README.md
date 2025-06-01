# Generators in Python

Generators are functions that return an iterator and allow for **lazy evaluation**, meaning values are produced one at a time only as needed. This makes them ideal for working with large data streams or infinite sequences.

## Core Concepts

- `yield` pauses and resumes function execution
- `next()` fetches the next value
- Generators are memory-efficient
- Generator expressions are like lazy list comprehensions
- Behind the scenes: `__iter__` and `__next__` protocol

## Files

| File                              | Description                                            |
| --------------------------------- | ------------------------------------------------------ |
| `generator_basics.py`             | Yielding values and `next()`                           |
| `generator_memory_efficiency.py`  | Compare memory usage between list vs generator         |
| `generator_fibonacci.py`          | Example: Fibonacci generator                           |
| `generator_expression_vs_list.py` | Memory difference: generator expression vs list comp   |
| `generator_custom_iterable.py`    | Custom generator class using `__iter__` and `__next__` |

## Run Example

```bash
python generator_basics.py
```

Docs: https://docs.python.org/3/library/stdtypes.html#generator-types
