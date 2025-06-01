## 5. How Python Internally Uses Descriptors

Descriptors are not just a feature for advanced developers—they are the **foundation** of many core Python features. Python internally uses descriptors to implement:

- `@property`
- `@staticmethod`
- `@classmethod`
- `super()`
- Method binding for instance methods

### Behind the Scenes: `@property`

The built-in `property()` function creates a **data descriptor** with optional getter, setter, and deleter methods. Here’s a simplified implementation:

```python
class Property:
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc or (fget.__doc__ if fget else None)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not self.fget:
            raise AttributeError("Unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("Can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if not self.fdel:
            raise AttributeError("Can't delete attribute")
        self.fdel(obj)

    def getter(self, fget): return type(self)(fget, self.fset, self.fdel, self.__doc__)
    def setter(self, fset): return type(self)(self.fget, fset, self.fdel, self.__doc__)
    def deleter(self, fdel): return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

Usage:

```python
class A:
    def __init__(self, value):
        self._value = value

    @Property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

x = A("Hello")
print(x.value)   # Calls __get__
x.value = "Hi"   # Calls __set__
```

### Method Binding with `__get__`

In Python, **instance methods** are descriptors too! When you access a method from an instance (e.g., `obj.method`), Python retrieves the function object from the class and calls its `__get__` method to bind it to the instance.

```python
class A:
    def method(self):
        return "called"

a = A()
bound_method = A.__dict__["method"].__get__(a, A)
print(bound_method())  # Output: called
```

### `@staticmethod` and `@classmethod`

These decorators are implemented using descriptor classes.

#### StaticMethod (non-data descriptor):

```python
class StaticMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        return self.func
```

#### ClassMethod (data descriptor):

```python
class ClassMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        return self.func.__get__(objtype, objtype)
```

Python wraps functions in these descriptor classes when you use `@staticmethod` or `@classmethod`.

### `super()` is Descriptor-Aware

The `super()` object has a custom `__getattribute__()` that invokes descriptors of the next class in the MRO (Method Resolution Order).

Example:

```python
class A:
    def greet(self): print("Hello from A")

class B(A):
    def greet(self): print("Hello from B")

class C(B):
    def greet(self):
        super().greet()  # calls B.greet()

C().greet()
```

The attribute lookup via `super()` ends up calling `B.__dict__['greet'].__get__(obj, C)` if it’s a descriptor.

### Descriptor Lookup is Built into `__getattribute__`

Python’s `__getattribute__()` (in both `object` and `type`) contains the logic for checking whether an attribute is a descriptor.

Simplified version:

```python
def __getattribute__(self, name):
    value = object.__getattribute__(self, name)
    if hasattr(value, '__get__'):
        return value.__get__(self, type(self))
    return value
```

This explains why just assigning a descriptor to a class field is enough for Python to start treating it differently—no magic or metaclasses required.

### Recap

| Feature         | Uses Descriptor? | Descriptor Type     |
| --------------- | ---------------- | ------------------- |
| `@property`     | ✅ Yes           | Data descriptor     |
| Instance method | ✅ Yes           | Non-data descriptor |
| `@staticmethod` | ✅ Yes           | Non-data descriptor |
| `@classmethod`  | ✅ Yes           | Data descriptor     |
| `super()`       | ✅ Yes           | Custom descriptor   |
