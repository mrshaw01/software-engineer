# 02 - Class vs. Instance Attributes in Python

This chapter explores the distinction between **class attributes**, **instance attributes**, and the use of **static** and **class methods** in Python's object-oriented programming model.

## Topics Covered

| File                          | Description                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `class_vs_instance.py`        | Demonstrates difference in scope and behavior of class vs. instance attributes |
| `robot_laws.py`               | Stores shared robot laws using class attributes                                |
| `instance_counter.py`         | Uses class attribute to count instances created and deleted                    |
| `static_method_example.py`    | Introduces static methods and their purpose                                    |
| `class_method_example.py`     | Explains how class methods differ from static and instance methods             |
| `class_method_inheritance.py` | Shows advantage of classmethods in inheritance scenarios                       |
| `person_counter.py`           | Real-world example of classmethod usage to track population count              |

## Key Concepts

### Class Attribute

- Shared across all instances.
- Defined directly in the class body.

### Instance Attribute

- Unique to each object.
- Defined within `__init__`.

### Static Method

- Doesn’t access class or instance data.
- Use for utility functions tied to a class context.

### Class Method

- Takes `cls` as the first parameter.
- Can access class state; useful in inheritance and factory patterns.

## Learning Tip

- Use class attributes for global counters or shared constants.
- Use instance attributes to track data specific to each object.
- Prefer class methods when functionality must vary across subclasses.
