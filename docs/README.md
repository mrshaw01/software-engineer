## Documentation

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
