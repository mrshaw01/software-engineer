# Python Descriptors

## 1. Introduction

Descriptors are a foundational feature in Python that allow developers to customize attribute access on objects. Introduced in **Python 2.2** as part of the _new-style class_ model, descriptors form the underlying mechanism for several built-in constructs like `@property`, `@staticmethod`, `@classmethod`, and even `super()`.

> A _descriptor_ is any object that defines one or more of the following methods:
> `__get__()`, `__set__()`, and `__delete__()`.
> If an object implements any of these methods, it is recognized as a descriptor.

The descriptor protocol gives programmers fine-grained control over how attributes are retrieved, modified, or deleted. This enables **managed attributes**—those whose access behavior can be intercepted and customized.

### Why Descriptors?

Before descriptors, attribute access was direct: reading or writing to `obj.attr` simply interacted with the object's `__dict__`. Descriptors allow this process to be intercepted and redirected through methods, enabling features like:

- Automatically computed values
- Validation logic on attribute assignment
- Method binding behavior (used in instance methods and class methods)

### Descriptor Lookup Chain

When accessing `obj.ap`, Python follows a defined lookup chain:

1. Check if `'ap'` is in `obj.__dict__`
2. If not found, check `type(obj).__dict__`
3. If still not found, walk through the base classes of `type(obj)`
4. If found and the value is a descriptor (has `__get__`, `__set__`, or `__delete__`), Python invokes the appropriate method

This makes descriptors a core part of Python’s attribute resolution mechanism.

### Example: Attribute Lookup in Action

```python
class A:
    ca_A = "class attribute of A"
    def __init__(self):
        self.ia_A = "instance attribute of A instance"

class B(A):
    ca_B = "class attribute of B"
    def __init__(self):
        super().__init__()
        self.ia_B = "instance attribute of B instance"

x = B()
print(x.ia_B)  # instance attribute of B instance
print(x.ca_B)  # class attribute of B
print(x.ia_A)  # instance attribute of A instance
print(x.ca_A)  # class attribute of A
```

If we try to access an undefined attribute:

```python
print(x.non_existing)
```

We'll receive an `AttributeError`:

```text
AttributeError: 'B' object has no attribute 'non_existing'
```
