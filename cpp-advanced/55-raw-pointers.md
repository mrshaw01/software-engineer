## Overview of Raw Pointers

A **raw pointer** is a built-in C++ type that holds the **memory address** of another variable or dynamically allocated object. Syntax-wise, a pointer is declared using the `*` symbol:

```cpp
int* p = nullptr;
```

Unlike references, pointers:

- Can be reassigned to point elsewhere
- Can be null (`nullptr`)
- Support pointer arithmetic
- Can be used to allocate/deallocate heap memory manually

However, **raw pointers do not manage ownership**. This leads to potential errors like **memory leaks, dangling pointers, double deletions**, and **buffer overruns**, especially in complex or exception-prone code paths.

## Common Use Cases

### 1. **Dynamic Allocation (Owning Pointer)**

```cpp
MyClass* obj = new MyClass(); // Allocated on heap
obj->print();
delete obj; // Must be called to free memory
```

**Pitfall**: Failing to `delete` causes a memory leak. Calling `delete` twice results in undefined behavior.

### 2. **Accessing Existing Memory (Non-owning Pointer)**

```cpp
int val = 42;
int* p = &val;
std::cout << *p << std::endl; // prints 42
```

Useful for performance when avoiding object copies. But pointer invalidation (e.g., via out-of-scope stack variables) is a risk.

### 3. **Pointer Arithmetic (Arrays)**

```cpp
int arr[] = {1, 2, 3};
int* p = arr;
*(p + 1) = 20;
std::cout << arr[1] << std::endl; // prints 20
```

Pointer arithmetic is prone to out-of-bounds access. Safer alternatives include range-based loops and STL containers.

## Best Practices for Raw Pointer Usage

| Situation                                 | Preferred Alternative                           |
| ----------------------------------------- | ----------------------------------------------- |
| Ownership of dynamically allocated memory | `std::unique_ptr`, `std::shared_ptr`            |
| Traversing a container                    | Iterators (`begin()`, `end()`), range-for loops |
| Function callbacks                        | `std::function`, lambdas                        |
| Polymorphic access                        | `std::unique_ptr<Base>` or reference semantics  |

Avoid raw pointers unless:

- You are implementing a low-level system component (e.g., allocator, driver)
- You require **explicit control over memory layout or performance**
- You can **guarantee correct ownership and lifetime discipline**

## Code Examples and Explanations

### 1. **Safe Initialization and Dereferencing**

```cpp
int* p = nullptr; // avoids garbage address
int value = 10;
p = &value;
int result = *p; // dereference to get 10
```

**Output**:

```
10
```

**Note**: Dereferencing an uninitialized or null pointer is undefined behavior.

### 2. **Danger of Double Delete**

```cpp
int* p = new int(5);
int* q = p;

delete p;
// delete q; // Undefined behavior: double delete
```

**Solution**: Use `std::shared_ptr` if shared ownership is required.

### 3. **Pointer and Object Semantics**

```cpp
class MyClass {
public:
    int val;
    MyClass(int v) : val(v) {}
};

void modify_by_pointer(MyClass* p) {
    p->val = 20;
}

void modify_by_value(MyClass obj) {
    obj.val = 30;
}

int main() {
    MyClass* ptr = new MyClass(10);
    modify_by_pointer(ptr);
    std::cout << ptr->val << std::endl; // 20

    modify_by_value(*ptr);
    std::cout << ptr->val << std::endl; // still 20

    delete ptr;
}
```

## Pointer to `void`

A `void*` pointer is a generic memory pointer without type information. Used primarily in low-level or C interop:

```cpp
void process(void* data, size_t length) {
    char* p = static_cast<char*>(data);
    for (size_t i = 0; i < length; ++i)
        *(p + i) = 0;
}
```

**Modern C++ Warning**: Use `void*` only when absolutely necessary (e.g., hardware buffers, C APIs). Avoid in application logic.

## Pointers to Functions

A raw function pointer allows behavior to be passed around, albeit without type safety or easy capturing of context.

```cpp
std::string greet(std::string name) {
    return "Hello, " + name;
}

std::string execute(std::string input, std::string(*func)(std::string)) {
    return func(input);
}

int main() {
    std::cout << execute("World", greet) << std::endl;
}
```

**Modern Replacement**: `std::function` or lambdas:

```cpp
auto f = [](std::string name) { return "Hi " + name; };
std::function<std::string(std::string)> fn = f;
```

## Summary: Raw Pointers – When and How

### Do Use Raw Pointers:

- In low-level systems code (e.g., embedded, drivers)
- Where precise memory control is required
- When implementing custom smart pointers or allocators

### Prefer Safer Alternatives:

- `std::unique_ptr` / `std::shared_ptr` for ownership
- `std::function`, lambdas for callbacks
- References for object access when nullability is not needed
- Containers and iterators for collection traversal
