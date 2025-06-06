# Function Caching in Python

Function caching is an optimization technique that stores the results of expensive function calls and reuses the cached result when the same inputs occur again. It’s especially effective for:

- Recursive or computationally expensive functions
- Functions involving I/O (e.g., reading files, calling APIs)
- Pure functions where output depends only on input

## 1. Why Use Function Caching?

Benefits of caching:

- Avoids redundant computation
- Reduces latency for repeated operations
- Minimizes load on external systems (e.g., databases, web services)

Common use cases:

- Recursive algorithms (e.g., Fibonacci)
- Configuration loaders
- API calls with idempotent results
- Expensive data processing functions

## 2. Built-in Caching with `functools.lru_cache` (Python 3.2+)

Python 3.2 introduced `functools.lru_cache`, which provides an easy-to-use decorator to cache results using a **Least Recently Used (LRU)** strategy.

### Basic Usage

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def slow_function(x):
    print(f"Computing slow_function({x})...")
    return x * x
```

- `maxsize`: the number of recent calls to cache
- If `maxsize=None`, the cache is unlimited (not recommended for unbounded input)

### Example: Fibonacci Sequence

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print([fib(n) for n in range(10)])
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

## 3. Managing the Cache

You can inspect or clear the cache at runtime:

```python
fib.cache_info()    # CacheInfo(hits=8, misses=10, maxsize=32, currsize=10)
fib.cache_clear()   # Clears all cached values
```

## 4. Manual Memoization (Pre-Python 3.2)

For older versions of Python, or for more control, you can create a custom caching decorator:

```python
from functools import wraps

def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### Caveats

- Only works with **hashable** arguments (e.g., `int`, `str`, `tuple`)
- Mutable types (`list`, `dict`, etc.) are **not supported** as keys

## 5. Mutable Argument Workaround

Neither `lru_cache` nor manual `memoize` work with unhashable types. You can convert them to immutable equivalents:

```python
@lru_cache(maxsize=64)
def process_data(data_tuple):
    return sum(data_tuple)

data = [1, 2, 3]
result = process_data(tuple(data))
```

## 6. Alternatives for Advanced Caching

If you need caching with expiration, disk storage, or more advanced policies:

| Library               | Features                               |
| --------------------- | -------------------------------------- |
| `functools.lru_cache` | Built-in LRU caching                   |
| `cachetools`          | TTL, LFU, LRU caching, custom policies |
| `joblib.Memory`       | Persistent disk caching (scikit-learn) |
| `diskcache`           | Disk + memory hybrid cache             |

### Example with `cachetools`:

```python
from cachetools import cached, LRUCache

cache = LRUCache(maxsize=100)

@cached(cache)
def compute(x):
    return x * 2
```

## 7. Pitfalls to Avoid

1. **Caching functions with side effects** (e.g., writing to a file, changing state)
2. **Caching based on dynamic input** (e.g., time-dependent or random values)
3. **Over-caching**: unbounded cache may lead to high memory usage
4. **Incorrect keying**: unhashable arguments will cause exceptions

## 8. Comparing Strategies

| Feature                         | `lru_cache` | Manual `memoize`   |
| ------------------------------- | ----------- | ------------------ |
| Built-in support                | ✅ Yes      | ❌ No              |
| Eviction strategy               | ✅ LRU      | ❌ None            |
| Configurable max size           | ✅ Yes      | ❌ No              |
| Works with unhashable           | ❌ No       | ❌ No              |
| Easy to clear                   | ✅ Yes      | ❌ Manual required |
| Introspection (`.cache_info()`) | ✅ Yes      | ❌ No              |

## 9. Summary

- Use `functools.lru_cache` for easy and efficient caching in Python 3.2+.
- Use manual memoization in older Python or when you need custom control.
- For caching with expiration or persistence, use libraries like `cachetools` or `joblib`.
- Always test caching behavior carefully for correctness, especially with mutable or dynamic inputs.
