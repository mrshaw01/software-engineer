# 20. The 'ABC' of Abstract Base Classes

This tutorial teaches how to use **Abstract Base Classes (ABCs)** in Python using the `abc` module. Abstract classes help enforce method definitions across subclasses and are useful in interface design and consistent APIs.

## What You'll Learn

- What abstract classes are and why they're useful.
- How to use the `abc` module and `@abstractmethod` decorator.
- How Python prevents instantiation of abstract classes.
- How abstract methods can still have base implementations.

## File Overview

| File                            | Description                                                             |
| ------------------------------- | ----------------------------------------------------------------------- |
| `01_non_abstract_class.py`      | Demonstrates a normal base class with no enforcement.                   |
| `02_abstract_class_error.py`    | Shows how instantiation fails when abstract methods aren't implemented. |
| `03_abstract_class_correct.py`  | Shows correct implementation of all abstract methods.                   |
| `04_abstract_with_base_impl.py` | Demonstrates base implementation using `super()`.                       |

## Run Examples

```bash
python 01_non_abstract_class.py
python 02_abstract_class_error.py
python 03_abstract_class_correct.py
python 04_abstract_with_base_impl.py
```
