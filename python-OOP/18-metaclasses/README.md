# 18. Metaclasses in Python

## Introduction

A **metaclass** is a class whose instances are classes. It allows you to control the behavior of classes just as classes control the behavior of instances. Python supports metaclasses through the `type` metaclass, and also allows you to define custom ones.

## Why Use Metaclasses?

Some practical uses of metaclasses include:

- Logging and profiling
- Interface checking
- Auto-registering classes
- Injecting methods/properties
- Enforcing naming conventions
- Singleton pattern
- Auto-synchronization and more

## Hierarchy Recap

- Instances are created from **Classes**
- Classes are created from **Metaclasses**

## Key Concepts

- The default metaclass in Python is `type`
- You can override `__new__` and/or `__init__` in a metaclass to modify class creation
- You use `metaclass=YourMetaClass` in a class definition to hook it to a custom metaclass

## Example Files

| File                          | Description                                                     |
| ----------------------------- | --------------------------------------------------------------- |
| `01_little_meta.py`           | Shows how to hook into class creation using a minimal metaclass |
| `02_essential_answers.py`     | Injects method into classes conditionally using a metaclass     |
| `03_singleton_metaclass.py`   | Implements Singleton pattern using a metaclass                  |
| `04_singleton_inheritance.py` | Singleton pattern without metaclasses using inheritance         |
| `05_singleton_decorator.py`   | Singleton pattern using a class decorator                       |
| `06_camelcase_metaclass.py`   | Converts CamelCase methods to snake_case using metaclass        |
| `07_camelcase_decorator.py`   | Same behavior as above, but using a decorator                   |

Run each script to see the metaclass magic in action!
