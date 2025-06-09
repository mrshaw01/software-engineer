# C++: Basics and Syntax

## Overview

This section introduces you to the fundamentals of C++ programming, including syntax, output, comments, variables, constants, and practical examples.

## 1. C++ Syntax

### Example

```cpp
#include <iostream>
using namespace std;

int main() {
  cout << "Hello World!";
  return 0;
}
```

### Explanation

- `#include <iostream>`: Includes the input/output stream library to use `cout`.
- `using namespace std;`: Allows direct access to `std` namespace objects like `cout`.
- `int main() { ... }`: The main function where the program starts.
- `cout << "Hello World!";`: Outputs text to the console.
- `return 0;`: Ends the `main` function.

**Note:** C++ is case-sensitive and every statement ends with a semicolon.

## 2. Omitting `using namespace std`

You can skip the `using namespace std;` line by prefixing standard objects with `std::`.

### Example

```cpp
#include <iostream>

int main() {
  std::cout << "Hello World!";
  return 0;
}
```

## 3. C++ Output (Print Text)

Use `cout` and the `<<` operator to print text or variables:

### Example

```cpp
cout << "Hello World!";
cout << "I am learning C++";
```

## 4. C++ Comments

### Single-line Comments

```cpp
// This is a single-line comment
cout << "Hello World!"; // Comment after code
```

### Multi-line Comments

```cpp
/* This is a
multi-line comment */
cout << "Hello World!";
```

## 5. C++ Variables

Variables are containers for storing data values.

### Basic Types

- `int`: Integer (e.g., `123`)
- `double`: Floating-point number (e.g., `19.99`)
- `char`: Character (e.g., `'a'`)
- `string`: Text (e.g., `"Hello"`)
- `bool`: Boolean (`true` or `false`)

### Declaration and Initialization

```cpp
int myNum = 15;
cout << myNum;
```

### Assign Later

```cpp
int myNum;
myNum = 15;
```

### Overwriting Values

```cpp
int myNum = 15;
myNum = 10;
cout << myNum; // Outputs 10
```

## 6. Other Data Types

```cpp
int myNum = 5;
double myFloatNum = 5.99;
char myLetter = 'D';
string myText = "Hello";
bool myBoolean = true;
```

## 7. Display Variables

```cpp
int myAge = 35;
cout << "I am " << myAge << " years old.";
```

```cpp
string name = "John";
int age = 35;
double height = 6.1;
cout << name << " is " << age << " years old and " << height << " feet tall.";
```

## 8. Add Variables Together

```cpp
int x = 5;
int y = 6;
int sum = x + y;
cout << sum;
```

## 9. Declare Multiple Variables

### Declare and Initialize

```cpp
int x = 5, y = 6, z = 50;
cout << x + y + z;
```

### Assign Same Value

```cpp
int x, y, z;
x = y = z = 50;
cout << x + y + z;
```

## 10. C++ Identifiers

Variable naming rules:

- Begin with a letter or `_`
- Contain letters, digits, and underscores
- Case-sensitive (`myVar` ≠ `myvar`)
- No spaces or special characters
- Avoid reserved keywords (`int`, `return`, etc.)

### Example

```cpp
int minutesPerHour = 60;
int m = 60; // Less descriptive
```

## 11. Constants

Use `const` to define a read-only variable:

```cpp
const int myNum = 15;
// myNum = 10; // Error!
```

Must be initialized when declared:

```cpp
const int minutesPerHour = 60;
```

## 12. Practical Examples

### Student Data

```cpp
int studentID = 15;
int studentAge = 23;
float studentFee = 75.25;
char studentGrade = 'B';

cout << "Student ID: " << studentID << "\n";
cout << "Student Age: " << studentAge << "\n";
cout << "Student Fee: " << studentFee << "\n";
cout << "Student Grade: " << studentGrade << "\n";
```

### Rectangle Area Calculator

```cpp
int length = 4;
int width = 6;
int area = length * width;

cout << "Length is: " << length << "\n";
cout << "Width is: " << width << "\n";
cout << "Area of the rectangle is: " << area << "\n";
```
