# How would you design a Python API for a high-performance library to be both user-friendly and efficient?

### 1. **Pythonic and Intuitive Interface**

- Use clear, consistent naming aligned with Python standards (e.g., `snake_case`).
- Avoid surprises: API should behave predictably (e.g., no expensive computation on simple property access).
- Follow established idioms from NumPy/SciPy so users feel at home.

**Example:**

```python
result = linalg.matmul(A, B)  # Mirrors numpy.linalg
```

### 2. **Minimize Python Overhead**

- Avoid Python-level loops for large data; favor vectorized operations.
- Perform all heavy computation in C++ or optimized low-level libraries (BLAS, LAPACK, or custom kernels).
- Use batch processing to reduce frequency of Python ↔ C++ boundary crossings.

### 3. **Leverage NumPy Compatibility**

- Accept and return **NumPy arrays** directly for zero-copy integration.
- Respect NumPy’s data layout and dtypes (e.g., `float32`, `float64`, `contiguous` vs `strided`).
- Use the **buffer protocol** or NumPy C-API to avoid unnecessary memory copies.

### 4. **Documentation and Usability**

- Provide docstrings, usage examples, and clear error messages.
- Allow optional keyword arguments for advanced users but sensible defaults for beginners.

### 5. **Extensibility and Safety**

- Modular design: separate compute from API bindings.
- Validate inputs early (e.g., shape checks) to prevent downstream crashes.
- Hide internal C++ complexity behind clean Python interfaces.

### 🔧 Tools I’d Use:

- **pybind11** or **Cython** for C++ integration
- **NumPy** for data structures and compatibility
- Possibly **Pydantic** or custom validators for type checking (if needed)

### Example Design Skeleton

```python
# python_api/linalg.py
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication (C = A @ B)."""
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible dimensions")
    return _cpp_backend.matmul(a, b)  # pybind11 call
```

### Summary

> A well-designed Python API should _feel native to Python developers_, _hide C++ complexity_, and _optimize performance-critical paths_ under the hood using compiled code and vectorization.
