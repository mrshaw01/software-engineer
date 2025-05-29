# How can you interface Python with high-performance C++ code?

### 1. **Use `pybind11` (Most Modern and Preferred)**

- **Header-only C++11 library** that makes exposing C++ functions and classes to Python very easy.
- Automatically handles type conversions (e.g., `std::vector` ↔ `list`, `Eigen::Matrix` ↔ `numpy.ndarray`).
- Minimal boilerplate, great for modern C++.

**Example:**

```cpp
// cpp_backend.cpp
#include <pybind11/pybind11.h>
int add(int a, int b) { return a + b; }

PYBIND11_MODULE(cpp_backend, m) {
    m.def("add", &add);
}
```

```python
# In Python
import cpp_backend
cpp_backend.add(2, 3)
```

### 2. **Use `Cython`**

- A hybrid language that looks like Python but compiles to C/C++.
- More control than pybind11 for performance tuning.
- Great for gradually accelerating Python code.

**Example:**

```cython
# add.pyx
cdef int add(int a, int b):
    return a + b
```

### 3. **Use Python C API (Low-Level, Powerful)**

- Full control over Python ↔ C interaction, but very verbose.
- Used in critical projects like CPython internals and NumPy core.

**Example:** Using `PyObject`, `PyArg_ParseTuple`, etc. — not ideal for rapid development.

### 4. **Other Alternatives**

- **SWIG**: Automatically generates bindings but harder to fine-tune.
- **ctypes** / **CFFI**: For interfacing with C (not C++); good for quick-and-dirty prototypes but not ideal for high-performance, complex codebases.

### 5. **Expose NumPy-Compatible APIs**

- When working with numerical data, expose functions that accept `PyArrayObject*` or `np.ndarray` via `pybind11` or NumPy C-API.
- This avoids conversions and improves performance.

### 6. **Build and Install**

- Use **CMake** or **setuptools + scikit-build** to compile and install your C++ code as a Python extension.

### Summary:

> The most common and robust way to interface Python with high-performance C++ is using **`pybind11`**, which offers clean syntax, efficient type conversion, and excellent NumPy support. For finer control or legacy integration, tools like **Cython** or the **Python C API** may be more appropriate.
