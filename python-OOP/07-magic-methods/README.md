# 07 - Magic Methods

This chapter introduces Python's "magic methods" or "dunder methods" (double underscore methods), such as `__init__`, `__add__`, `__repr__`, `__str__`, and more.

Magic methods enable operator overloading and customization of standard behavior. For example, you can define how objects of your class respond to `+`, `==`, `str()`, `repr()` and even function calls using `__call__`.

## Highlights

- `__init__`: Constructor called on object creation.
- `__str__`: Returns user-friendly string representation.
- `__repr__`: Returns unambiguous string (used by developers).
- `__add__`, `__radd__`: Overload `+` and reverse `+`.
- `__iadd__`: Overload `+=`.
- `__mul__`, `__rmul__`, `__imul__`: Overload multiplication.
- `__call__`: Make an object callable like a function.

## Examples

### Length class

Supports addition of different units:

```python
from length import Length as L
print(L(2.56,"m") + L(3,"yd") + L(7.8,"in") + L(7.03,"cm"))
```

### Currency class

Supports addition and multiplication of currency values:

```python
from currency_converter import Ccy

x = Ccy(10.00, "EUR")
y = Ccy(10.00, "GBP")
print(x + y)        # currency-aware addition
print(2 * x + y * 0.9)  # scalar multiplication
```
