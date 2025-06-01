## 3. Data Descriptors vs Non-Data Descriptors

Descriptors in Python fall into two main categories based on which methods of the descriptor protocol they implement:

| Type                    | Implements                              | Takes Precedence Over Instance Attr? | Common Use Case                 |
| ----------------------- | --------------------------------------- | ------------------------------------ | ------------------------------- |
| **Data Descriptor**     | `__get__` + (`__set__` or `__delete__`) | ✅ Yes                               | Validated or managed attributes |
| **Non-Data Descriptor** | `__get__` only                          | ❌ No                                | Methods, computed values        |

### ⚙️ How Precedence Works

Python’s attribute lookup order gives priority in the following order:

1. **Data descriptor** in class (calls `__get__`)
2. **Instance attribute** in `obj.__dict__`
3. **Non-data descriptor** in class (calls `__get__`)
4. **Fallback**: class attribute or `__getattr__()` if defined

This precedence makes **data descriptors** useful for enforcing constraints or managing internal state—such as read-only attributes, or validation of values on assignment.

### Example: Non-Data Descriptor

A descriptor that only defines `__get__` acts like a _computed attribute_, but allows instance attributes to override it:

```python
class NonDataDescriptor:
    def __get__(self, instance, owner):
        return "Descriptor result"

class MyClass:
    attr = NonDataDescriptor()

obj = MyClass()
print(obj.attr)  # Descriptor result

obj.attr = "Overridden"
print(obj.attr)  # Overridden (instance attribute takes precedence)
```

### Example: Data Descriptor (Read-only)

By defining `__set__` that raises an error, we can create a read-only attribute:

```python
class ReadOnly:
    def __init__(self, value):
        self._value = value

    def __get__(self, instance, owner):
        return self._value

    def __set__(self, instance, value):
        raise AttributeError("This attribute is read-only")

class Demo:
    const = ReadOnly(42)

d = Demo()
print(d.const)     # 42
d.const = 99       # ❌ Raises AttributeError
```

### Real-World Analogy

- **Data descriptors** are like locked doors: you must go through the gatekeeper (`__set__`, `__get__`, `__delete__`) to access or modify the value.
- **Non-data descriptors** are like signboards: they show default behavior unless someone posts a new sign (instance variable), which overrides the display.

### Visual Lookup Chain

```text
Access obj.attr:

If type(obj).__dict__['attr'] is a data descriptor:
    → Call attr.__get__(obj, type(obj))
Else if 'attr' in obj.__dict__:
    → Return obj.__dict__['attr']
Else if type(obj).__dict__['attr'] is a non-data descriptor:
    → Call attr.__get__(obj, type(obj))
Else if 'attr' in type(obj).__dict__:
    → Return type(obj).__dict__['attr']
Else:
    → Fallback to __getattr__(attr)
```
