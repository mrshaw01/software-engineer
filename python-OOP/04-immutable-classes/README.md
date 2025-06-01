# 04 - Creating Immutable Classes in Python

This chapter introduces immutable class design in Python. Immutability means that an object’s state cannot be modified after it is created. Immutable objects are common in functional programming, and Python supports several patterns to create them.

## Why Immutability?

- ✅ **Thread-safe**: No data races or lock management.
- **Predictable**: The object state stays consistent.
- **Cacheable**: Supports efficient reuse and memoization.
- **Simpler tests**: No mutable state means fewer side effects.
- **Easier debugging**: No hidden mutations to track.
- **Prevents unintended changes**.
- **Hashable**: Suitable for `set` and `dict` keys.
- **Functional programming style**: Promotes purity and composability.

## Techniques for Creating Immutable Classes

| File                      | Technique Description                                        |
| ------------------------- | ------------------------------------------------------------ |
| `getter_only.py`          | Getter-only methods, no setters                              |
| `property_readonly.py`    | Read-only properties using `@property`                       |
| `dataclass_frozen.py`     | `@dataclass(frozen=True)` to auto-generate immutable classes |
| `namedtuple_immutable.py` | Using `collections.namedtuple`                               |
| `slots_example.py`        | `__slots__` to prevent dynamic attribute creation            |

## Tip

Prefer `@dataclass(frozen=True)` or `namedtuple` for lightweight immutable models. Use `__slots__` for memory optimization, not immutability enforcement.
