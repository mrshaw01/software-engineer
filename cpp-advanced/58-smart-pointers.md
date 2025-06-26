## 1. **Why Smart Pointers?**

In C++, dynamically allocated resources (`new`, `malloc`, file handles, etc.) must be explicitly released, which leads to:

- Memory/resource leaks when `delete` is forgotten
- Double deletion
- Dangling pointers
- Exception-unsafety due to early exits

**Smart pointers** solve these problems through **automatic ownership** and **destructor-based cleanup**, leveraging the RAII idiom.

## 2. **Overview of Standard Smart Pointers**

Header: `<memory>`

### 2.1 `std::unique_ptr<T>`

- **Exclusive ownership** of a dynamically allocated object.
- Non-copyable but **moveable**.
- Best suited as the default smart pointer.

```cpp
#include <memory>

void use_unique_ptr() {
    std::unique_ptr<int> uptr(new int(42));
    *uptr += 1;
    std::cout << *uptr << std::endl; // Output: 43
} // Automatically deletes the integer
```

#### Safer alternative (C++14 and above):

```cpp
auto uptr = std::make_unique<int>(42);
```

### 2.2 `std::shared_ptr<T>`

- **Shared ownership** with internal reference counting.
- The object is destroyed when the last `shared_ptr` owning it is destroyed.

```cpp
std::shared_ptr<std::string> sp1 = std::make_shared<std::string>("hello");
std::shared_ptr<std::string> sp2 = sp1; // Shared ownership
std::cout << sp1.use_count() << std::endl; // Output: 2
```

**Performance tip**: Prefer `make_shared<T>()` to reduce allocation overhead (one allocation instead of two).

### 2.3 `std::weak_ptr<T>`

- Non-owning observer of a `shared_ptr`.
- Used to **break circular dependencies**.

```cpp
std::shared_ptr<int> sp = std::make_shared<int>(10);
std::weak_ptr<int> wp = sp;

if (auto locked = wp.lock()) {
    std::cout << *locked << std::endl; // 10
}
```

## 3. **Smart Pointer Behavior and Operators**

- All standard smart pointers overload `*` and `->` for pointer-like access.
- Use `.get()` to retrieve the raw pointer (e.g., for legacy interop).
- Use `.reset()` to explicitly release ownership.
- Use `.release()` (only on `unique_ptr`) to relinquish ownership without deleting the object.

```cpp
std::unique_ptr<int> uptr = std::make_unique<int>(99);
int* raw = uptr.release(); // uptr no longer manages memory
delete raw; // manual deletion required
```

## 4. **Exception Safety and RAII**

Consider:

```cpp
void process() {
    std::unique_ptr<Resource> r(new Resource());
    throw std::runtime_error("failure"); // `r` still safely destroyed
}
```

Smart pointers ensure deterministic cleanup, making code **strongly exception-safe** with minimal boilerplate.

## 5. **Avoiding Pitfalls**

### 5.1 **Never pass `new T()` directly to function parameters**

```cpp
// Risky: resource may leak if exception thrown before transfer
process_resource(std::unique_ptr<T>(new T));

// Safer:
auto resource = std::make_unique<T>();
process_resource(std::move(resource));
```

### 5.2 **Avoid raw pointer access**

Only use `.get()` when interacting with APIs that do not support smart pointers.

### 5.3 **Don't use `shared_ptr` unnecessarily**

Use `shared_ptr` only when ownership is genuinely shared; otherwise prefer `unique_ptr`.

## 6. **Polymorphism with Smart Pointers**

Smart pointers support polymorphic deletion when used with virtual destructors.

```cpp
struct Base {
    virtual ~Base() = default;
};

struct Derived : Base {};

std::unique_ptr<Base> ptr = std::make_unique<Derived>();
```

The base class **must** have a virtual destructor to avoid undefined behavior during destruction.

## 7. **Comparison Table**

| Feature            | `unique_ptr`          | `shared_ptr`               | `weak_ptr`                |
| ------------------ | --------------------- | -------------------------- | ------------------------- |
| Ownership          | Exclusive             | Shared (ref-counted)       | Observes shared ownership |
| Copyable           | No                    | Yes                        | Yes                       |
| Moveable           | Yes                   | Yes                        | Yes                       |
| Thread-safe        | No                    | Yes (control block only)   | Yes                       |
| Reference counting | No                    | Yes                        | Yes (but non-owning)      |
| Best use case      | Default for ownership | Shared ownership scenarios | Break cycles / observers  |
| Size (on 64-bit)   | 8 bytes               | 16 bytes                   | 8 bytes                   |

## 8. **Smart Pointers with Containers**

Smart pointers can be stored in STL containers:

```cpp
std::vector<std::unique_ptr<MyClass>> vec;
vec.push_back(std::make_unique<MyClass>());
```

This allows dynamic storage with automatic cleanup, without raw pointers.

## 9. **Custom Deleters**

Smart pointers can be customized to clean up non-memory resources (e.g., file handles).

```cpp
std::unique_ptr<FILE, decltype(&fclose)> fptr(fopen("file.txt", "r"), &fclose);
```

Custom deleters are especially useful for:

- OS handles
- OpenGL/Vulkan resources
- Network sockets

## 10. **Legacy and COM Smart Pointers (Windows)**

When dealing with COM interfaces, use ATL smart pointers:

- `CComPtr<T>` for automatic `AddRef/Release`
- `CComQIPtr<T>` for `QueryInterface` support

These are specialized for COM, not suitable for standard C++ object ownership.

## 11. **Summary and Best Practices**

| Rule                           | Best Practice                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------- |
| Ownership                      | Use `unique_ptr` unless shared ownership is required                            |
| Creation                       | Use `make_unique` or `make_shared` for safety and efficiency                    |
| Avoid `new`                    | Except in custom allocators or legacy wrappers                                  |
| RAII                           | Always encapsulate resources in RAII types (smart pointers, file streams, etc.) |
| Avoid `.get()`                 | Unless interacting with raw-pointer legacy APIs                                 |
| Prefer composition             | Avoid smart pointers for objects with deterministic lifetime via composition    |
| Use `weak_ptr` to break cycles | Especially in graphs, trees, or parent-child structures with back-pointers      |

## Closing Thoughts

Smart pointers represent one of the core tools of **modern C++ resource management**. Their design aligns with C++'s deterministic destruction model, enabling safe, expressive, and high-performance code. Transitioning from raw pointers to smart pointers is a critical step toward writing maintainable, exception-safe, and robust systems. Every modern C++ codebase should default to smart pointers unless low-level optimization or interoperability explicitly requires otherwise.
