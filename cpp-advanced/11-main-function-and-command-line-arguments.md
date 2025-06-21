# Main Function and Command-Line Arguments in C++

## Overview

In every C++ program, the `main` function marks the entry point of execution. It is essential and must be defined; otherwise, the compiler will produce an error. The only exceptions are for dynamic-link libraries (DLLs) and static libraries, which do not require a `main` function.

Before entering `main`, all static members without initializers are zero-initialized. On Microsoft compilers, global static objects are also initialized before `main` is executed.

## Rules Specific to `main`

The `main` function has several restrictions:

- It **cannot be overloaded**
- It **cannot be declared `inline`**
- It **cannot be declared `static`**
- Its **address cannot be taken**
- It **cannot be explicitly called** from within your program

## Function Signatures

C++ allows two standard signatures for the `main` function:

```cpp
int main();
int main(int argc, char* argv[]);
```

If no return value is provided explicitly, the compiler assumes a default return value of `0`.

### Parameters

- `argc`: Number of command-line arguments, including the program name (always ≥ 1).
- `argv`: Array of C-style strings. `argv[0]` is the name of the executable, `argv[1]` to `argv[argc - 1]` are command-line arguments, and `argv[argc]` is `NULL`.

## Platform-Specific Behavior (Windows/Microsoft)

### `wmain` and `_tmain`

- `wmain`: Wide-character version of `main`, used when the program supports Unicode.

  ```cpp
  int wmain();
  int wmain(int argc, wchar_t* argv[]);
  ```

- `_tmain`: A macro that resolves to `main` or `wmain` based on whether `_UNICODE` is defined.

### Returning `void` from `main`

Microsoft allows `main` and `wmain` to return `void`, though it's not standard. If so, returning an exit code must be done via the `exit()` function.

### Environment Parameter (`envp`)

Some compilers support this extended signature:

```cpp
int main(int argc, char* argv[], char* envp[]);
int wmain(int argc, wchar_t* argv[], wchar_t* envp[]);
```

- `envp` is an array of environment variables (null-terminated strings).
- It remains a static snapshot—modifying the environment later does not change `envp`.

## Example: Displaying Environment Variables

```cpp
#include <iostream>
#include <string.h>

int main(int argc, char* argv[], char* envp[]) {
    bool numberLines = false;
    if (argc == 2 && _stricmp(argv[1], "/n") == 0) {
        numberLines = true;
    }

    for (int i = 0; envp[i] != NULL; ++i) {
        if (numberLines) std::cout << i << ": ";
        std::cout << envp[i] << "\n";
    }
}
```

## Command-Line Parsing Behavior (Windows/MSVC)

Parsing rules used by Microsoft C/C++:

1. Arguments are delimited by whitespace (space or tab).
2. `argv[0]` is the program name and may contain quoted strings.
3. A quoted string is treated as a single argument.
4. A backslash before a double quote affects parsing based on the number of backslashes.

### Examples

| Command-Line Input | `argv[1]` | `argv[2]` | `argv[3]` |
| ------------------ | --------- | --------- | --------- |
| `"abc" d e`        | abc       | d         | e         |
| `a\\b d"e f"g h`   | a\b       | de fg     | h         |
| `a\\\"b c d`       | a"b       | c         | d         |
| `a\\\\"b c" d e`   | a\b c     | d         | e         |
| `a"b"" c d`        | ab" c d   |           |           |

## Example: Argument Display

```cpp
#include <iostream>

int main(int argc, char* argv[], char* envp[]) {
    std::cout << "\nCommand-line arguments:\n";
    for (int i = 0; i < argc; ++i)
        std::cout << "  argv[" << i << "]   " << argv[i] << "\n";
}
```

## Wildcard Expansion

By default, wildcards like `*` or `?` are **not** expanded in `argv`. To enable this behavior:

- Use `setargv.obj` for `main`
- Use `wsetargv.obj` for `wmain`

Add these to your `/link` options during compilation.

## Suppressing Command-Line and Environment Processing

To minimize runtime overhead, you can disable processing:

- Use `noarg.obj` to disable argument parsing
- Use `noenv.obj` to disable environment parsing

Only do this if your program never uses command-line arguments or environment variables. Do **not** disable environment processing if your program uses `spawn` or `exec` family functions.

## Summary

- `main` is the entry point of any C++ program.
- Use `argc`, `argv`, and optionally `envp` to process command-line and environment data.
- Microsoft provides extended support with `wmain`, `_tmain`, and wildcard handling.
- Understand quoting and backslash rules when writing cross-platform command-line tools.
