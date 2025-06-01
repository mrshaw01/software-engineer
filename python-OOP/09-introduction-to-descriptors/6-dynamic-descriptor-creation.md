## 6. Dynamic Descriptor Creation and Best Practices

Descriptors don’t have to be manually coded into your class—you can also **generate them dynamically at runtime**, giving you extreme flexibility for metaprogramming and frameworks (e.g., ORM fields, serializers, etc.).

### ⚙️ Dynamically Adding Properties at Runtime

This example demonstrates how to add a property with getter and setter methods **at runtime**, based on a string name:

```python
class DynPropertyClass:
    def add_property(self, attr_name):
        """Dynamically add a property to the class"""
        def getter(self):
            return getattr(self, f"_{type(self).__name__}__{attr_name}")

        def setter(self, value):
            setattr(self, f"_{type(self).__name__}__{attr_name}", value)

        # Assign to the class
        setattr(
            type(self),
            attr_name,
            property(fget=getter, fset=setter, doc="Auto-generated property")
        )

# Example usage
x = DynPropertyClass()
x.add_property('name')
x.add_property('city')

x.name = "Henry"
x.city = "Hamburg"

print(x.name, x.city)
print(x.__dict__)  # See the internal mangled attribute names
```

📌 **Note:** While this pattern is powerful, it can make debugging and maintenance harder. Use it with care in framework-like systems.

## ✅ Descriptor Best Practices

### 1. **Use built-in `@property` unless you need extra control**

- For simple managed attributes, `@property` is cleaner and more readable.

### 2. **Use `WeakKeyDictionary` for per-instance state**

- Descriptors are shared at the class level, so per-instance data must be stored separately. `WeakKeyDictionary` avoids memory leaks.

```python
from weakref import WeakKeyDictionary
```

### 3. **Don’t define descriptors in `__init__`**

- They must be assigned as **class attributes** to trigger the descriptor protocol.

### 4. **Prefer data descriptors when you need override protection**

- Data descriptors take precedence over instance attributes.

### 5. **Avoid side effects in `__get__`**

- Keep `__get__` fast and side-effect free if possible. Lazy evaluation and caching are fine, but logging, I/O, or mutation may be surprising.

## ⚠️ Common Pitfalls

| Pitfall                                              | Explanation                                                                            |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Defining descriptors in `__init__`                   | They won't trigger the descriptor protocol—they must be part of the class body.        |
| Using instance variables inside descriptors directly | Can cause shared state across instances—use `WeakKeyDictionary` instead.               |
| Overriding `__getattribute__` without care           | Breaks descriptor behavior or introduces bugs. Always call `super().__getattribute__`. |
| Not accounting for `obj` being `None` in `__get__`   | Happens when accessed via the class, e.g., `Class.attr`.                               |

## 🧵 Summary

Descriptors provide the foundation for many core Python features and enable highly customizable behavior for attribute access. They are ideal when you need:

- Validated or constrained attributes
- Lazy-loaded or cached values
- Logging, type checking, or access control
- Properties that require internal storage logic
