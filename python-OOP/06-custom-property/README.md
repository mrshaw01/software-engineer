# 06 - Implementing a Custom Property Class

In this chapter, we go deeper into how Python's built-in `property` works by reimplementing our own version. This is especially useful for gaining a better understanding of descriptors and decorators in Python.

## ✨ Why Build a Custom Property?

- Understand how `@property` works under the hood
- Learn about Python’s descriptor protocol
- Get comfortable with decorator patterns
- Useful for creating debug-friendly or logging-enabled properties

## 📘 Files Included

| File                 | Description                                   |
| -------------------- | --------------------------------------------- |
| `our_property.py`    | Minimal implementation of `property` class    |
| `chatty_property.py` | A verbose version of `property` for debugging |

## 💡 Key Concepts

- `__get__`, `__set__`, and `__delete__` implement descriptor behavior
- Custom `getter`, `setter`, and `deleter` methods replicate `@property` chaining
- Demonstrates how decorators return new descriptors with updated behavior
