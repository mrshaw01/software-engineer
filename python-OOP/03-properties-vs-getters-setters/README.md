# 03 - Properties vs. Getters and Setters in Python

This chapter covers the evolution from traditional getter/setter methods to Pythonic property-based encapsulation. It highlights the simplicity and flexibility that Python properties offer, and explains when getter/setter methods may still be useful.

## 📘 Topics Covered

| File                             | Description                                                              |
|----------------------------------|--------------------------------------------------------------------------|
| `getter_setter_basic.py`        | Traditional encapsulation with explicit `get_` and `set_` methods        |
| `public_attribute.py`           | Pythonic class design with public attributes                             |
| `clamped_setter.py`             | Uses setter validation to clamp values into a range                      |
| `property_decorator.py`         | Uses `@property` and `@x.setter` decorators for Pythonic encapsulation   |
| `property_alt_syntax.py`        | Alternative property declaration using `property()`                      |
| `robot_condition_property.py`   | Property depending on multiple internal values                           |
| `convert_to_property.py`        | Migrating from public attribute to property without interface breaking   |
| `generic_getattr_setattr.py`    | Uses `__getattr__` and `__setattr__` for generalized property behavior    |
| `conditional_setattr.py`        | Advanced example with conditional validation in `__setattr__`            |
| `getter_setter_use_case.py`     | When to prefer traditional getters/setters (e.g. additional arguments)   |

## 🔍 Key Takeaways

- **Use properties** for Pythonic, simple, and readable interfaces.
- **Use private attributes with properties** when encapsulation or validation is needed.
- **Fallback to getter/setter methods** when external compatibility or additional arguments are required.
- Properties preserve interface compatibility when implementation changes.

