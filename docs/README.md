## Documentation

### Questions

1. How would you design a Python API for a high-performance library to be both user-friendly and efficient?
2. What steps can you take to minimize Python overhead in numeric computations?
3. How would you ensure your Python API integrates well with existing scientific Python libraries (NumPy/SciPy)?
4. How can you interface Python with high-performance C++ code?
5. What are the challenges of calling into C++ from Python, and how do you mitigate them?
6. How do you handle Python’s Global Interpreter Lock (GIL) when using a C++ backend?
7. Explain the difference between CUDA cores and Tensor Cores on NVIDIA GPUs.
8. How does memory coalescing work in CUDA, and why is it important?
9. Describe the concept of warp divergence and its impact on GPU performance.
10. What is the role of shared memory in CUDA kernels?
11. Have you used OpenCL, and how does it compare to CUDA?
12. Why are BLAS and LAPACK important for numerical computing in Python (e.g. NumPy/SciPy)?
13. How would you benchmark and optimize a computational kernel (e.g., a matrix operation) on CPU vs GPU?
14. Give an example of a numerical stability issue and how to address it.
15. How do memory access patterns affect performance on modern CPUs?
16. How does NumPy achieve performance that pure Python can’t?
17. What is CuPy, and how is it related to NumPy?
18. Have you used SciPy, and how does it improve on NumPy for certain tasks?
19. What is Numba and how can it speed up Python code?
20. Discuss your experience with PyTorch or TensorFlow and how they achieve high performance.
21. What is JAX and how does it enable high-performance Python computations?
22. Do you have experience with parallel computing frameworks or libraries (e.g. MPI, Dask)?

### Keywords

- `frozenset`
- `namedtuple`, `@dataclass(frozen=True)`, `__slots__`
- Decorators:
  - [`@functools.wraps`](https://docs.python.org/3/library/functools.html#functools.wraps): Preserves metadata when wrapping functions
  - [`@functools.update_wrapper`](https://docs.python.org/3/library/functools.html#functools.update_wrapper)
- Generators: `yield`
- `Parameters`: Define expected inputs in functions/methods
- `Arguments`: Provide actual inputs during function calls
- `Attributes`: Store object state (data)
- `Properties`: Control access to attributes with logic
- [`@property`](https://docs.python.org/3/library/functions.html#property)
- Context managers: `__enter__`, `__exit__`

### Attribute Naming Conventions in Python

| Naming   | Type      | Meaning                                                                 |
| -------- | --------- | ----------------------------------------------------------------------- |
| `name`   | Public    | Accessible inside or outside the class.                                 |
| `_name`  | Protected | Should be accessed only within the class or subclasses (by convention). |
| `__name` | Private   | Name mangled to prevent access from outside the class.                  |

### Method Types in Python

| Feature            | Instance Method                    | Class Method                                           | Static Method                      |
| ------------------ | ---------------------------------- | ------------------------------------------------------ | ---------------------------------- |
| Decorator          | _(none)_                           | `@classmethod`                                         | `@staticmethod`                    |
| First Argument     | `self` (instance)                  | `cls` (class)                                          | None                               |
| Access to Instance | ✅ Yes                             | ❌ No                                                  | ❌ No                              |
| Access to Class    | ✅ Via `self.__class__` (indirect) | ✅ Yes via `cls`                                       | ❌ No                              |
| Use Case           | Operates on object state           | Operates on class state, often used as factory methods | Utility function, no state needed  |
| Bound To           | Instance                           | Class                                                  | Class                              |
| Example Call       | `obj.method()`                     | `Class.method()` or `obj.method()`                     | `Class.method()` or `obj.method()` |
