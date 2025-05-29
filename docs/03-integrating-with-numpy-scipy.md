# How would you ensure your Python API integrates well with existing scientific Python libraries (NumPy/SciPy)?

### 1. **Accept and Return NumPy Arrays Directly**

- Design the API to **accept `np.ndarray` inputs** and return `np.ndarray` outputs.
- This ensures zero-friction interoperability with NumPy, SciPy, Matplotlib, etc.

```python
def my_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _cpp_backend.dot(a, b)
```

### 2. **Avoid Data Copies (Zero-Copy Integration)**

- Use the **buffer protocol** or **NumPy’s C API** to operate on raw memory of NumPy arrays without copying.
- Handle **strided** and **non-contiguous** arrays properly or explicitly require `np.ascontiguousarray()` when needed.

### 3. **Respect NumPy Dtypes and Memory Layout**

- Support standard `dtype`s like `float32`, `float64`, `int32`, etc.
- Be aware of row-major (C-style) layout and alignment for performance.
- Avoid breaking assumptions about array shape, type, or layout unless clearly documented.

### 4. **Use Familiar Semantics**

- Match the **behavior and naming** of NumPy/SciPy functions (e.g., broadcasting rules, axis arguments, default values).
- Avoid surprising behaviors like expensive operations in property access.

### 5. **Support NumPy Protocols**

- Implement or support protocols like:

  - `__array__()` for implicit casting
  - `__array_function__()` and `__array_ufunc__()` for NumPy dispatching if applicable

This allows advanced interoperability and override capability.

### 6. **Document Clearly and Consistently**

- Follow NumPy/SciPy-style documentation conventions (numpydoc).
- Include `Parameters`, `Returns`, and `Examples` sections.

### 7. **Test Compatibility**

- Write tests that combine your library with SciPy workflows, e.g.,:

  - Feed your output into `scipy.optimize`, `scipy.linalg`, or `matplotlib`.
  - Use `np.testing.assert_allclose` to validate compatibility.

### 8. **Optional: Expose ufuncs**

- If applicable, register your functions as **universal functions (ufuncs)** for seamless NumPy integration and broadcasting.

### Summary:

> Integrating well with NumPy/SciPy means **speaking their language**: operate on NumPy arrays directly, follow their conventions, and avoid unnecessary data copies. This ensures your API plays nicely with the broader scientific Python ecosystem.
