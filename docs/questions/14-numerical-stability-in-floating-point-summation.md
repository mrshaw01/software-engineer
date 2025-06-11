# Give an example of a numerical stability issue and how to address it.

A common example of a **numerical stability issue** is **summing a large list of floating-point numbers**. When using naive summation in floating-point arithmetic, **small rounding errors accumulate**, leading to inaccurate results.

### Problem: **Naive Floating-Point Summation**

```python
# Example: summing numbers of vastly different magnitudes
data = [1e8, 1, -1e8]
total = sum(data)  # result is 0.0, but expected is 1.0
```

This is due to **catastrophic cancellation**: adding a large number and a small number can **lose precision** in the small value.

### Solution 1: **Kahan Summation Algorithm**

Kahan’s algorithm keeps a **running compensation** for lost low-order bits:

```python
def kahan_sum(data):
    total = 0.0
    c = 0.0  # compensation
    for x in data:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total
```

This yields **much more accurate results** for long or ill-conditioned sums.

### Solution 2: **Order the Summation**

Sort numbers by magnitude (smallest first) to reduce error accumulation.

```python
total = sum(sorted(data, key=abs))
```

### Other Examples of Stability Issues

| Problem                              | Stable Alternative                                  |
| ------------------------------------ | --------------------------------------------------- |
| **Matrix inversion**                 | Solve `Ax = b` with `LU` or `QR` factorization      |
| **Subtracting nearly equal numbers** | Use algebraic reformulation or higher precision     |
| **Division by small values**         | Add small epsilon (`+1e-8`) to denominator          |
| **Finite-difference derivatives**    | Use central differences or symbolic differentiation |

### Summary:

> A classic numerical stability issue is **catastrophic cancellation during summation**. Techniques like **Kahan summation** or **reordering operands** mitigate error. More broadly, **stable algorithms avoid operations prone to magnifying roundoff errors**, such as direct inversion, and instead rely on **factorization**, **regularization**, or **compensated computation**.
