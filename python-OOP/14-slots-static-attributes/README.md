# 14. Slots: Avoiding Dynamically Created Attributes

This tutorial demonstrates how to use `__slots__` in Python to prevent dynamically adding attributes to objects, and to reduce memory usage for classes that are instantiated many times.

## Why Use `__slots__`?

By default, Python uses a `__dict__` to store instance attributes, allowing dynamic attribute creation. However, this flexibility comes with a memory cost. When creating many instances, this cost becomes significant.

Using `__slots__`, we define a static structure that:

- Saves memory by preventing dynamic attribute creation.
- Improves performance when creating large numbers of objects.

## Example: Without `__slots__`

```python
class A:
    pass

a = A()
a.x = 66
a.y = "dynamically created attribute"

print(a.__dict__)  # {'x': 66, 'y': 'dynamically created attribute'}
```
