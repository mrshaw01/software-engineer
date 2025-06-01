## 4. Practical Use Cases of Descriptors

Descriptors are powerful tools that allow developers to control attribute behavior at a fine-grained level. They underpin many Python features—including `property()`, `staticmethod()`, and `classmethod()`—but can also be used to implement custom logic for:

- **Validation and constraints**
- **Lazy loading / caching**
- **Type enforcement**
- **Computed properties**
- **Logging or access tracing**

### 1. Validation Descriptor (Voting Age)

Use case: Only allow assignment of valid values (e.g., enforce minimum age).

```python
from weakref import WeakKeyDictionary

class Voter:
    required_age = 18
    def __init__(self):
        self._ages = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self._ages.get(instance)

    def __set__(self, instance, age):
        if age < self.required_age:
            raise ValueError(f"{instance.name} is too young to vote.")
        self._ages[instance] = age
        print(f"{instance.name} is eligible to vote.")

class Person:
    voter_age = Voter()
    def __init__(self, name, age):
        self.name = name
        self.voter_age = age

p = Person("Alice", 22)  # ✅
p = Person("Bob", 16)    # ❌ Raises ValueError
```

### 2. Computed Property (Emulate `@property`)

Use case: Define dynamic, computed attributes.

```python
class Celsius:
    def __init__(self, temperature=0):
        self._temperature = temperature

    def get_temp(self):
        print("Getting temperature...")
        return self._temperature

    def set_temp(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        print("Setting temperature...")
        self._temperature = value

    temperature = property(get_temp, set_temp)

c = Celsius(25)
print(c.temperature)   # Getting temperature...
c.temperature = 30     # Setting temperature...
```

### 3. Caching Descriptor

Use case: Cache the result of an expensive computation after the first access.

```python
class CachedProperty:
    def __init__(self, func):
        self.func = func
        self._cache = {}

    def __get__(self, instance, owner):
        if instance not in self._cache:
            print("Computing value...")
            self._cache[instance] = self.func(instance)
        return self._cache[instance]

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @CachedProperty
    def area(self):
        from math import pi
        return pi * self.radius ** 2

c = Circle(10)
print(c.area)  # Computes once
print(c.area)  # Uses cache
```

### 4. Debugging Descriptor

Use case: Log all access to a field for auditing or debugging.

```python
class DebugDescriptor:
    def __init__(self, name):
        self.name = name
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        value = self.data.get(instance)
        print(f"Accessing {self.name} = {value}")
        return value

    def __set__(self, instance, value):
        print(f"Setting {self.name} = {value}")
        self.data[instance] = value

class Demo:
    x = DebugDescriptor("x")

d = Demo()
d.x = 42
print(d.x)
```

### 5. Type Enforcement Descriptor

Use case: Automatically check the type of assigned values.

```python
class Typed:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be of type {self.expected_type}")
        self.data[instance] = value

class Point:
    x = Typed("x", float)
    y = Typed("y", float)

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1.0, 2.5)  # ✅
p.x = "not a float"  # ❌ Raises TypeError
```
