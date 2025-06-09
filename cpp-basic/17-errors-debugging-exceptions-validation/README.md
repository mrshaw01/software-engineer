# C++: Errors, Debugging, Exceptions, and Input Validation

This section covers how to handle **errors**, use **debugging techniques**, implement **exception handling**, and perform **input validation** in C++.

Understanding and fixing problems is a key part of programming. This chapter will help you build confidence by identifying common issues and learning how to prevent or resolve them.

## Errors

### Compile-Time Errors

These errors stop your code from compiling. Common examples:

- **Missing semicolon**:

  ```cpp
  int x = 5
  ```

  Error:

  ```
  error: expected ',' or ';' before 'cout'
  ```

- **Using undeclared variable**:

  ```cpp
  cout << myVar;
  ```

  Error:

  ```
  error: 'myVar' was not declared in this scope
  ```

- **Mismatched types**:

  ```cpp
  int x = "Hello";
  ```

  Error:

  ```
  error: invalid conversion from 'const char*' to 'int'
  ```

### Runtime Errors

These occur after the code compiles but crashes or behaves unexpectedly.

- **Divide by zero**:

  ```cpp
  int result = a / b;  // If b is 0
  ```

- **Accessing out-of-bounds array element**:

  ```cpp
  cout << numbers[8];  // Invalid if array size < 9
  ```

- **Using deleted memory (dangling pointer)**:

  ```cpp
  delete ptr;
  cout << *ptr;  // Dangerous
  ```

### Good Habits

- Always initialize variables
- Use meaningful names
- Keep code clean and organized
- Keep functions short
- Use loops and conditions correctly
- Read compiler messages carefully

## Debugging

Debugging is the process of finding and fixing problems in your code.

### 1. Print Debugging

Use `cout` to print values during execution:

```cpp
cout << "Before division\n";
int z = x / y;
cout << "After division\n";  // Might not print if error occurs
```

### 2. Check Variable Values

```cpp
int result = x - y;
cout << "Result: " << result << "\n";
```

### 3. Use a Debugger

IDEs like Visual Studio and VS Code support:

- Breakpoints
- Step-by-step execution
- Variable watches

### 4. Understand Error Messages

C++ error messages often point directly to the problem:

```
error: expected ';' before 'return'
```

## Exception Handling

Exceptions let you handle unexpected situations at runtime using `try`, `throw`, and `catch`.

### Example

```cpp
try {
  throw 505;
} catch (int errorCode) {
  cout << "Error occurred: " << errorCode;
}
```

### Real-Life Example

```cpp
try {
  int age = 15;
  if (age < 18) throw age;
  cout << "Access granted.";
} catch (int myNum) {
  cout << "Access denied. Age: " << myNum;
}
```

### Catch All Types

```cpp
catch (...) {
  cout << "Some error occurred.";
}
```

## Input Validation

Input validation ensures that user input is acceptable before processing.

### Validate Integer Input

```cpp
int number;
while (!(cin >> number)) {
  cout << "Invalid input. Try again: ";
  cin.clear();
  cin.ignore(10000, '\n');
}
```

### Validate Range

```cpp
do {
  cin >> number;
} while (number < 1 || number > 5);
```

### Validate Text

```cpp
string name;
do {
  getline(cin, name);
} while (name.empty());
```
