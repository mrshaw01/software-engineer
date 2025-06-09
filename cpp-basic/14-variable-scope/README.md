# C++: Variable Scope

## What is Scope?

**Scope** refers to where a variable can be accessed or used in a program. In C++, variables are only available in the region of code where they are defined.

There are two main types of scope:

- **Local scope**
- **Global scope**

## Local Scope

A variable declared **inside a function** has **local scope**. It can only be used within that function:

```cpp
void myFunction() {
  int x = 5;  // Local variable
  cout << x;
}
```

Trying to use `x` outside `myFunction()` will result in an error:

```cpp
void myFunction() {
  int x = 5;
}

int main() {
  myFunction();
  cout << x;  // Error: x is not declared in this scope
  return 0;
}
```

## Global Scope

A variable declared **outside any function** has **global scope** and can be accessed from any function in the program:

```cpp
int x = 5;  // Global variable

void myFunction() {
  cout << x << "\n";
}

int main() {
  myFunction();
  cout << x;
  return 0;
}
```

## Naming Conflicts Between Local and Global Variables

If you declare a **local variable with the same name** as a global variable, the local one will **hide** the global one within its scope:

```cpp
int x = 5;  // Global variable

void myFunction() {
  int x = 22;  // Local variable
  cout << x << "\n";  // Prints 22 (local x)
}

int main() {
  myFunction();
  cout << x;  // Prints 5 (global x)
  return 0;
}
```

⚠️ **Avoid using the same name** for global and local variables to reduce confusion and bugs.

## Modifying Global Variables

Global variables can be **modified inside any function**, which may lead to unexpected behavior:

```cpp
int x = 5;  // Global variable

void myFunction() {
  cout << ++x << "\n";  // Increments x to 6
}

int main() {
  myFunction();
  cout << x;  // Outputs 6
  return 0;
}
```

## Conclusion

- Use **local variables** wherever possible. They make your code easier to understand and maintain.
- Use **global variables** with caution, as they can be changed from anywhere, making code harder to debug.
- Understanding scope helps prevent naming conflicts and improves the reliability of your programs.
