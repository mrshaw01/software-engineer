## 2. Descriptor Protocol

Descriptors define how attribute access is managed in Python. They allow developers to intercept and control what happens when attributes are accessed, assigned, or deleted.

At the heart of this mechanism lies the **descriptor protocol**, which consists of three optional methods:

```python
descr.__get__(self, obj, type=None) -> value
descr.__set__(self, obj, value) -> None
descr.__delete__(self, obj) -> None
```

If a class implements any of these methods, instances of that class become **descriptors**.

### Data vs. Non-Data Descriptors

- **Non-data descriptor**: Implements only `__get__()`. Commonly used for bound methods.
- **Data descriptor**: Implements `__set__()` or `__delete__()` (or both). These descriptors take precedence over instance attributes during lookup.

Example: A simple data descriptor:

```python
class SimpleDescriptor:
    def __init__(self, initval=None):
        print("__init__ called with:", initval)
        self.__set__(self, initval)

    def __get__(self, instance, owner):
        print("Getting self.val:", self.val)
        return self.val

    def __set__(self, instance, value):
        print("Setting self.val to:", value)
        self.val = value

class MyClass:
    x = SimpleDescriptor("green")

m = MyClass()
print(m.x)        # Triggers __get__
m.x = "yellow"    # Triggers __set__
print(m.x)
```

### Accessing Descriptors and Lookup Rules

Descriptor methods are called **automatically** by Python when an attribute is accessed, modified, or deleted. They reside in the _class dictionary_ of the owning class, not the instance.

- If `x` is a class attribute that is a descriptor, then `x.__get__(instance, owner)` is triggered when accessed.
- This only works if the descriptor is defined in the class, not in the instance's `__init__`.

Let’s examine how Python stores descriptor references:

```python
print(m.__dict__)            # → {}
print(MyClass.__dict__)      # 'x' is in here as a descriptor
print(SimpleDescriptor.__dict__)  # Contains the __get__ and __set__ methods
```

Descriptor lookup has a specific priority order:

1. **Data descriptor** in class → calls `__get__`
2. **Instance attribute** (in `__dict__`)
3. **Non-data descriptor** → calls `__get__`
4. `__getattr__` fallback (if defined)

### How Python Uses Descriptors Internally

Python's internal method `__getattribute__()` (implemented in C) manages attribute access. It checks whether an object defines any descriptor methods and, if so, delegates to them:

```python
def __getattribute__(self, key):
    val = type.__getattribute__(self, key)
    if hasattr(val, '__get__'):
        return val.__get__(None, self)
    return val
```

Example invocation:

```python
m.__getattribute__("x")
# Calls SimpleDescriptor.__get__
```

### Custom Descriptor with Validation (Voting Age)

```python
from weakref import WeakKeyDictionary

class Voter:
    required_age = 18

    def __init__(self):
        self.age = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.age.get(instance)

    def __set__(self, instance, value):
        if value < self.required_age:
            raise Exception(f"{instance.name} is not old enough to vote.")
        self.age[instance] = value
        print(f"{instance.name} can vote in Germany")

    def __delete__(self, instance):
        del self.age[instance]

class Person:
    voter_age = Voter()
    def __init__(self, name, age):
        self.name = name
        self.voter_age = age
```

Usage:

```python
p1 = Person("Ben", 23)
p2 = Person("Emilia", 22)
print(p2.voter_age)
```

### A Custom Property Class Using Descriptors

This emulates Python’s built-in `property()`:

```python
class Property:
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc or (fget.__doc__ if fget else None)

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        if not self.fget: raise AttributeError("Unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if not self.fset: raise AttributeError("Can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if not self.fdel: raise AttributeError("Can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

### Runtime-Generated Properties

Properties can even be added dynamically at runtime:

```python
class DynPropertyClass:
    def add_property(self, attribute):
        def get_attr(self):
            return getattr(self, f"_{type(self).__name__}__{attribute}")
        def set_attr(self, value):
            setattr(self, f"_{type(self).__name__}__{attribute}", value)
        setattr(type(self), attribute, property(get_attr, set_attr))

x = DynPropertyClass()
x.add_property('name')
x.name = "Henry"
print(x.name)  # → Henry
```
