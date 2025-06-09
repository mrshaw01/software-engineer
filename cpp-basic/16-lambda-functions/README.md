# C++: Lambda Functions

## What is a Lambda Function?

A **lambda function** is a small, unnamed (anonymous) function that you can define directly in your code.
It's useful when you need a quick function without creating a full function declaration.

Think of it as a "function on the fly."

## Lambda Syntax

```cpp
[capture](parameters) {
  // code
};
```

- **`[]` (Capture clause)**: Lets the lambda use variables from the surrounding scope.
- **`(parameters)`**: Inputs just like any regular function.
- **`{}`**: Function body.

## Basic Lambda Example

```cpp
int main() {
  auto message = []() {
    cout << "Hello World!\n";
  };

  message();
  return 0;
}
```

**Output:**

```
Hello World!
```

## Lambda with Parameters

```cpp
int main() {
  auto add = [](int a, int b) {
    return a + b;
  };

  cout << add(3, 4);  // Outputs: 7
  return 0;
}
```

## Passing Lambdas to Functions

You can pass a lambda as an argument using `std::function`:

```cpp
#include <iostream>
#include <functional>
using namespace std;

void myFunction(function<void()> func) {
  func();
  func();
}

int main() {
  auto message = []() {
    cout << "Hello World!\n";
  };

  myFunction(message);
  return 0;
}
```

**Output:**

```
Hello World!
Hello World!
```

## Using Lambdas in Loops

```cpp
int main() {
  for (int i = 1; i <= 3; i++) {
    auto show = [i]() {
      cout << "Number: " << i << "\n";
    };
    show();
  }
  return 0;
}
```

**Output:**

```
Number: 1
Number: 2
Number: 3
```

## Capture Clause: `[ ]`

You can capture outside variables into a lambda:

### Capture by Value

```cpp
int main() {
  int x = 10;

  auto show = [x]() {
    cout << x;
  };

  x = 20;
  show();  // Outputs 10 (copy made at definition time)
  return 0;
}
```

### Capture by Reference

```cpp
int main() {
  int x = 10;

  auto show = [&x]() {
    cout << x;
  };

  x = 20;
  show();  // Outputs 20 (uses updated variable)
  return 0;
}
```

## Lambda vs Regular Functions

| Use Case               | Regular Function     | Lambda Function      |
| ---------------------- | -------------------- | -------------------- |
| Reused multiple times  | ✅ Yes               | ❌ Not ideal         |
| Long or complex logic  | ✅ Yes               | ❌ Avoid             |
| Short, one-time logic  | ❌ Verbose           | ✅ Perfect use case  |
| Passing into functions | ❌ More setup needed | ✅ Simple and inline |

### Example

**Regular Function**

```cpp
int add(int a, int b) {
  return a + b;
}
```

**Lambda Function**

```cpp
auto add = [](int a, int b) {
  return a + b;
};
```

## Summary

- Lambda functions provide a concise way to define small, inline functions.
- They can take parameters and use variables from outside (via capture).
- They are great for short-lived operations, especially as function arguments or inside loops.
- Use regular functions when reusability and clarity are more important.
