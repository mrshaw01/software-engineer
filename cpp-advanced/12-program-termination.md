# C++ Program Termination

This document outlines the standard ways a C++ program can terminate, along with detailed behavior around program shutdown, resource cleanup, and termination-related functions.

## Termination Methods

In C++, a program can be terminated using one of the following methods:

- `exit()` function
- `abort()` function
- `return` statement in `main`

## `exit()` Function

The `exit` function is declared in `<cstdlib>`. It terminates the program and returns a specified exit code to the operating system. Typical conventions:

- `EXIT_SUCCESS`: program completed successfully (typically zero)
- `EXIT_FAILURE`: program failed (non-zero)

Upon calling `exit`, the C++ runtime:

- Executes registered `atexit` handlers
- Destroys global and static objects (in reverse initialization order)

```cpp
#include <cstdlib>
int main() {
    exit(EXIT_SUCCESS);
}
```

## `abort()` Function

Declared in `<cstdlib>`, `abort` forcibly terminates the program without performing any cleanup:

- Does **not** destroy global/static objects
- Does **not** execute `atexit` handlers

This is useful in scenarios such as unrecoverable errors or debugging situations.

```cpp
#include <cstdlib>
int main() {
    abort();
}
```

> Note: On Windows, `abort()` may still trigger some termination routines for compatibility reasons.

## `atexit()` Function

`atexit` allows you to register cleanup functions to be called when `exit()` is invoked. These functions are called in **reverse order** of registration.

```cpp
#include <cstdlib>
#include <iostream>

void cleanup() {
    std::cout << "Cleaning up...\n";
}

int main() {
    atexit(cleanup);
    return 0;
}
```

Registered functions are **not** called if the program exits via `abort()`.

## `return` from `main`

Using `return` in `main` behaves similarly to calling `exit()` with the same value:

- Destroys local automatic variables in `main`
- Invokes `exit()` internally

```cpp
int main() {
    return 3; // Equivalent to exit(3);
}
```

If no return is specified, the compiler inserts `return 0` implicitly.

## Destruction of Static and Thread-Local Objects

When a C++ program terminates via `exit()` or `return` from `main`, the runtime performs the following cleanup:

1. Registered `atexit` handlers are executed.
2. Thread-local and static objects are destroyed **in reverse order of their initialization**.

This deterministic destruction ensures proper cleanup of file handles, memory, and other resources.

## Example: Static Object Cleanup

```cpp
#include <cstdio>

class ShowData {
public:
    ShowData(const char* name) { fopen_s(&file, name, "w"); }
    ~ShowData() { if (file) fclose(file); }
    void Disp(const char* msg) { fputs(msg, file); }
private:
    FILE* file = nullptr;
};

ShowData sd1("CON");
ShowData sd2("hello.dat");

int main() {
    sd1.Disp("hello to default device\n");
    sd2.Disp("hello to file hello.dat\n");
    return 0;
}
```

In this example:

- `sd1` and `sd2` are static global objects
- Their destructors are called after `main` returns
- File handles are properly closed

## Alternative: Block-Scoped Resource Ownership

```cpp
int main() {
    ShowData sd1("CON"), sd2("hello.dat");
    sd1.Disp("hello to default device\n");
    sd2.Disp("hello to file hello.dat\n");
    return 0;
}
```

By scoping resources inside `main`, they are destroyed as part of stack unwinding, ensuring localized and deterministic cleanup.

## Summary

| Method         | Cleanup Performed | Static Destructors Called | atexit Handlers | Custom Return Code |
| -------------- | ----------------- | ------------------------- | --------------- | ------------------ |
| `exit(code)`   | Yes               | Yes                       | Yes             | Yes                |
| `abort()`      | No                | No                        | No              | No                 |
| `return code;` | Yes               | Yes                       | Yes             | Yes                |

Prefer `return` or `exit()` for normal termination, and reserve `abort()` for abnormal termination paths where cleanup isn't guaranteed or desired.
