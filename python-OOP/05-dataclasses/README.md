# 05 - Dataclasses in Python

Dataclasses simplify the creation of classes that primarily store data. Introduced in Python 3.7, they automatically generate boilerplate methods such as `__init__`, `__repr__`, `__eq__`, and optionally `__hash__`.

## Benefits of Using Dataclasses

- ✅ Less boilerplate: Auto-generates `__init__`, `__repr__`, `__eq__`, and `__hash__`
- Ideal for immutable data with `frozen=True`
- Works well with `typing` for better static checks
- Supports default values and default factories
- Inheritance and customization-friendly

## Topics Covered

| File                       | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `traditional_class.py`     | Regular class with manual `__init__` and `__repr__` |
| `basic_dataclass.py`       | Basic `@dataclass` usage                            |
| `frozen_dataclass.py`      | Immutable `@dataclass(frozen=True)`                 |
| `traditional_immutable.py` | Manual implementation of an immutable class         |
| `dataclass_in_set_dict.py` | Hashable dataclasses used in sets and dictionaries  |
| `exercise_book.py`         | Exercise solution: Book class with dataclass        |

## Tip

Use `frozen=True` when your data should never change. This makes your instances hashable and usable in sets and dict keys.
