# Storage Classes in C++

In C++, **storage classes** define the _lifetime_, _visibility_, and _linkage_ of variables and functions. Understanding how these properties interact is critical to managing program state, memory efficiency, and encapsulation. The language provides four primary storage class specifiers: `static`, `extern`, `thread_local`, and `mutable`. This part focuses on the `static` keyword, which is arguably the most nuanced of the group.

### Overview of Storage Duration and Linkage

Each object in C++ has:

- **Storage duration**: Determines when memory is allocated and deallocated.
- **Linkage**: Determines whether multiple declarations refer to the same object across translation units.
- **Scope**: Where in the program the object can be referenced by name.

Unless otherwise specified, variables declared inside a block have **automatic storage duration**, **no linkage**, and **block scope**. This means memory is allocated on the stack when the block is entered and deallocated when it exits.

## `static` Keyword

The `static` storage class has multiple meanings based on context. It changes the **lifetime** of an object and may affect **linkage**. Its semantics differ depending on where it's applied:

### 1. **Local Static Variables**

When used inside a function or block scope:

- The variable is initialized only once.
- Its value persists across multiple invocations of the function.
- Lifetime: From the first invocation to the end of the program.

```cpp
void accumulate(int x) {
    static int sum = 0;
    sum += x;
    std::cout << "sum = " << sum << std::endl;
}
```

Each call to `accumulate()` updates the same `sum` variable. This is commonly used in counters, memoization, and logging.

### 2. **Static Variables at Global or Namespace Scope**

When used at global or namespace scope:

- The variable has **static storage duration** (lives for the entire program).
- The variable has **internal linkage**, meaning it's not accessible from other translation units.

```cpp
// file1.cpp
static int global_counter = 0;  // not visible to other files
```

This is a key tool for encapsulation at file level.

### 3. **Static Class Data Members**

When declared inside a class:

- The member is shared by all instances.
- It's defined outside the class definition exactly once.

```cpp
class MyClass {
public:
    static int counter;
};

int MyClass::counter = 0;
```

All instances access the same `counter` variable. This is commonly used for tracking instances or global class-wide settings.

### 4. **Static Member Functions**

Static functions inside a class:

- Do not have access to the implicit `this` pointer.
- Can be called without an instance.
- Are often used for utility-like functionality tied to the class.

```cpp
class Logger {
public:
    static void log(const std::string& msg) {
        std::cout << msg << std::endl;
    }
};

Logger::log("Start");  // valid
```

Such methods are stateless and class-global by nature.

### Special Cases and Rules

- Static variables inside **unions** are not allowed.
- **Const static integral members** can be initialized inside the class.
- A **global anonymous union** must be declared `static` to limit linkage.

## Static Initialization and Thread Safety

Since C++11, **static local variables** are initialized in a thread-safe manner:

- Initialization is guaranteed to occur once, even if multiple threads call the function concurrently.
- However, access to the variable **after** initialization must still be synchronized manually if it's mutable.

```cpp
void thread_safe_init() {
    static std::vector<int> data = {1, 2, 3};  // Initialized once safely
    // concurrent access to `data` still needs synchronization
}
```

This feature is known as _magic statics_, and can be disabled using `/Zc:threadSafeInit-` in MSVC for build reproducibility reasons.

## Deprecated and Removed Keywords

- The `register` keyword is deprecated in C++11 and **removed in C++17**. It has no effect on modern compilers.
- The `auto` keyword no longer specifies storage class as of C++11—it now refers to type inference.

## Summary

| Context                | Effect of `static`                                           |
| ---------------------- | ------------------------------------------------------------ |
| Inside a function      | Extends lifetime across function calls                       |
| Global or namespace    | Restricts linkage to current translation unit                |
| Inside class           | Shares member across all instances                           |
| Static member function | Can be called without object; cannot access instance members |

## `extern`

The `extern` keyword tells the compiler that a variable or function exists, but is defined elsewhere—usually in another translation unit. It provides **external linkage**, meaning the declared symbol can be used across multiple source files.

```cpp
// header.h
extern int global_counter;

// file1.cpp
#include "header.h"
int global_counter = 0;

// file2.cpp
#include "header.h"
void increment() { global_counter++; }
```

The compiler assumes the variable exists elsewhere and will be resolved during linking. `extern` is also used to declare variables defined in assembly or C libraries.

## `thread_local` (C++11)

The `thread_local` storage class ensures that each thread has its own instance of a variable. These variables are created when a thread starts and destroyed when it ends.

```cpp
thread_local int counter = 0;
```

Each thread operates on its own `counter`, isolated from others.

### Key Characteristics:

- **Scope**: `thread_local` can be applied to global variables, local static variables, and class static members.
- **Lifetime**: Tied to the lifetime of the thread, not the program.
- **Default Behavior**: At block scope, `thread_local` implies `static`.

```cpp
void log_id() {
    thread_local int id = generate_thread_id();
    std::cout << "Thread ID: " << id << std::endl;
}
```

Each thread gets a separate `id`, initialized independently.

### Restrictions and Notes:

- Not applicable to **function declarations** or **non-static class members**.
- Must be explicitly specified in both declaration and definition.
- Initialization of `thread_local` variables is **not guaranteed across threads in DLLs**.
- Should be avoided with `std::launch::async` due to unpredictable thread reuse.

## `register` (Deprecated in C++17)

The `register` keyword was historically used to suggest to the compiler that a variable should be stored in a CPU register for faster access. In modern C++, this specifier has **no effect** and is deprecated:

```cpp
register int x = 10; // No-op in modern C++, causes a warning in C++17
```

Modern compilers already perform aggressive register allocation during optimization, rendering `register` obsolete. It's retained as a reserved word for backward compatibility.

## Example: Automatic vs Static Initialization

The difference between automatic and static storage duration becomes evident when considering initialization timing and lifetime.

```cpp
class InitDemo {
public:
    InitDemo(const char* name) {
        std::cout << "Initializing: " << name << std::endl;
        szName = new char[strlen(name) + 1];
        strcpy(szName, name);
    }

    ~InitDemo() {
        std::cout << "Destroying: " << szName << std::endl;
        delete[] szName;
    }

private:
    char* szName;
};

int main() {
    InitDemo I1("Auto I1");
    {
        InitDemo I2("Auto I2");
        static InitDemo I3("Static I3");
    }
    std::cout << "Exited block.\n";
}
```

### Output:

```
Initializing: Auto I1
In block.
Initializing: Auto I2
Initializing: Static I3
Destroying: Auto I2
Exited block.
Destroying: Auto I1
Destroying: Static I3
```

### Analysis:

- `I1` and `I2` are local automatic objects: destroyed when their scope ends.
- `I3` is a static local object: constructed once when first encountered, and destroyed **at program termination**.
- Demonstrates deferred initialization: objects are not initialized until control reaches their definition.
- Highlights how static storage duration impacts object lifetime and order of destruction.

## Summary Table

| Storage Specifier | Scope        | Lifetime           | Linkage       | Use Case                                  |
| ----------------- | ------------ | ------------------ | ------------- | ----------------------------------------- |
| `static`          | Block/global | Entire program     | Internal      | Persist state, file encapsulation         |
| `extern`          | Global       | Defined elsewhere  | External      | Cross-file sharing of symbols             |
| `thread_local`    | Block/global | Lifetime of thread | Internal/None | Thread-specific state                     |
| `register`        | Deprecated   | N/A                | N/A           | Historical use only; no longer meaningful |
