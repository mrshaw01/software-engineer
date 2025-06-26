# Overview: `new` and `delete` Operators

C++ provides the `new` and `delete` operators to allocate and deallocate memory from the **free store** (commonly referred to as the heap). These operators not only manage raw memory but also invoke constructors and destructors, making them superior to C-style `malloc`/`free` in object-oriented code.

## 1. The `new` Operator

### Basic Syntax

```cpp
int* p = new int;           // Allocates and value-initializes an int
int* arr = new int[10];     // Allocates an array of 10 ints
MyClass* obj = new MyClass; // Calls MyClass() constructor
```

### Semantics

- Calls `operator new(size_t)` internally
- Allocates raw memory
- Constructs object(s) at the allocated location using placement new
- Returns a pointer of the appropriate type

### Example

```cpp
#include <iostream>

int main() {
    int* p = new int(42);
    std::cout << *p << std::endl; // Output: 42
    delete p;
}
```

### Exception Handling on Allocation Failure

By default, `new` throws `std::bad_alloc` if allocation fails:

```cpp
try {
    int* big = new int[1'000'000'000'000];
} catch (const std::bad_alloc& e) {
    std::cerr << "Allocation failed: " << e.what() << std::endl;
}
```

To avoid exceptions:

```cpp
int* big = new (std::nothrow) int[1'000'000'000'000];
if (!big) {
    std::cerr << "Allocation failed (nothrow)" << std::endl;
}
```

## 2. The `delete` Operator

### Basic Syntax

```cpp
delete p;       // For single objects
delete[] arr;   // For arrays
```

### Semantics

- Calls the destructor if the object is not trivially destructible
- Calls `operator delete(void*)`
- Frees the raw memory previously allocated by `new`

### Common Pitfalls

- **Undefined behavior** if `delete` is applied to a pointer not allocated by `new`
- **Mismatch**: using `delete` on memory allocated with `new[]`, or vice versa
- **Double delete**: calling `delete` more than once on the same pointer

### Example

```cpp
MyClass* obj = new MyClass();
delete obj;  // OK

int* arr = new int[10];
delete[] arr; // OK
```

## 3. Overloading `operator new` and `operator delete`

Classes can define custom memory allocation logic:

### Example: Custom Initialization

```cpp
#include <cstring>
#include <cstdlib>

class MyBuffer {
public:
    void* operator new(size_t size, char fill) {
        void* ptr = std::malloc(size);
        if (ptr) std::memset(ptr, fill, size);
        return ptr;
    }

    void operator delete(void* ptr) {
        std::free(ptr);
    }
};

int main() {
    MyBuffer* buf = new ('X') MyBuffer;
    delete buf;
}
```

**Explanation**: The `new ('X') MyBuffer` form passes `'X'` to the custom `operator new`, which fills the allocated memory with `'X'`.

## 4. `operator new[]` and `operator delete[]`

You can also overload array-specific versions:

```cpp
class MyClass {
public:
    static void* operator new[](size_t size) {
        return std::malloc(size);
    }

    static void operator delete[](void* ptr) {
        std::free(ptr);
    }
};
```

Used in:

```cpp
MyClass* objs = new MyClass[10];
delete[] objs;
```

## 5. Sized Deallocation (C++14 and Later)

For performance, the compiler can pass size information to `operator delete`:

```cpp
void operator delete(void* ptr, size_t size);
```

This allows allocators to more efficiently determine how to reclaim the memory block.

## 6. Memory Leak Detection: Tracking Allocations

```cpp
#include <iostream>
#include <cstdlib>

int allocations = 0;

void* operator new(std::size_t size) {
    ++allocations;
    std::cout << "Allocating " << size << " bytes (#" << allocations << ")\n";
    return std::malloc(size);
}

void operator delete(void* ptr) noexcept {
    --allocations;
    std::cout << "Deallocating (remaining: " << allocations << ")\n";
    std::free(ptr);
}
```

**Usage**: Track memory allocations and detect leaks in test or debug builds.

## 7. Best Practices

### ✔ Prefer Smart Pointers

Use `std::unique_ptr` and `std::shared_ptr` to automate memory management:

```cpp
std::unique_ptr<MyClass> obj = std::make_unique<MyClass>();
```

### ✔ Always Pair `new` with `delete`

Every dynamic allocation should have a corresponding deallocation:

```cpp
MyClass* ptr = new MyClass();
// ... use ptr
delete ptr;
```

Never forget `delete`, especially when `new` is used inside a loop or conditional block.

### ✘ Avoid Manual Memory Management in Modern Code

Only use `new`/`delete` if:

- You are implementing low-level libraries (e.g., custom allocators)
- Managing resources not handled by standard types (e.g., memory-mapped I/O)
- Working in environments without full STL support

## 8. Placement New

Placement new constructs objects in pre-allocated memory:

```cpp
#include <new>

char buffer[sizeof(MyClass)];
MyClass* obj = new (buffer) MyClass(); // Calls constructor in-place
obj->~MyClass(); // Must manually call destructor
```

Used in:

- Embedded systems
- Custom allocators
- In-place construction on shared memory

## Summary Table

| Operation                | Action                                 |
| ------------------------ | -------------------------------------- |
| `new T`                  | Allocates memory, constructs `T`       |
| `delete p`               | Calls destructor of `*p`, frees memory |
| `new T[n]`               | Allocates array, constructs elements   |
| `delete[] p`             | Destroys elements, deallocates array   |
| `operator new(size_t)`   | Raw allocation                         |
| `operator delete(void*)` | Raw deallocation                       |
| `new (nothrow) T`        | Returns nullptr on failure             |
| `new (buffer) T`         | Placement new                          |

## Conclusion

The `new` and `delete` operators provide the foundation for dynamic memory management in C++. While powerful, they require meticulous attention to detail to avoid undefined behavior, memory leaks, and security vulnerabilities. In modern C++, use them primarily when implementing resource management utilities, custom containers, or performance-critical systems.

For most applications, **favor standard abstractions** like smart pointers, containers, and allocators. When you must use `new` and `delete`, **enforce clear ownership**, **ensure matching pairs**, and **audit memory behavior carefully**. This discipline is essential to writing robust, performant, and secure C++ systems.
