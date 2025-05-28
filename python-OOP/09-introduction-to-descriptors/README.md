# 9. Introduction to Descriptors

This chapter explores **Python Descriptors**, the core mechanism behind properties, methods, static/class methods, and more. Descriptors let you customize how attributes are accessed or modified via special methods:

- `__get__(self, instance, owner)`
- `__set__(self, instance, value)`
- `__delete__(self, instance)`

## Contents

| File                               | Description                                                                |
| ---------------------------------- | -------------------------------------------------------------------------- |
| `01_attribute_lookup_chain.py`     | Demonstrates Python's attribute lookup chain using classes and inheritance |
| `02_simple_descriptor.py`          | Implements a basic data descriptor and demonstrates how it's triggered     |
| `03_data_vs_nondata_descriptor.py` | Explains the difference between data and non-data descriptors              |
| `04_voter_age_descriptor.py`       | Example using `WeakKeyDictionary` for per-instance attribute storage       |
| `05_custom_property.py`            | Full re-implementation of Python's built-in `property()` using descriptors |
| `06_robot_property_demo.py`        | Demonstrates property behavior with validation logic                       |
| `07_decorator_inside_class.py`     | Shows how decorators can be used to enhance method behavior                |
| `08_dynamic_property_generator.py` | Demonstrates how to generate descriptors dynamically at runtime            |

## Key Concepts

- **Non-Data Descriptor**: Only defines `__get__()`. Example: methods.
- **Data Descriptor**: Defines `__set__()` or `__delete__()`. Example: properties.
- **Lookup Chain**: Attributes are first checked in `obj.__dict__`, then `type(obj).__dict__`, then base classes.
- **Dynamic Descriptors**: Descriptors can be created at runtime using `setattr` and `property`.
