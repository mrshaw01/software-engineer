# C++: Operators

## 1. Introduction to Operators

Operators are special symbols in C++ used to perform operations on variables and values.

### Example

```cpp
int x = 100 + 50;
```

You can also combine variables and values:

```cpp
int sum1 = 100 + 50;        // 150
int sum2 = sum1 + 250;      // 400
int sum3 = sum2 + sum2;     // 800
```

C++ provides different groups of operators:

- Arithmetic Operators
- Assignment Operators
- Comparison Operators
- Logical Operators
- Bitwise Operators

## 2. Arithmetic Operators

Arithmetic operators are used for basic mathematical operations.

| Operator | Name           | Description                      | Example |
| -------- | -------------- | -------------------------------- | ------- |
| `+`      | Addition       | Adds two values                  | `x + y` |
| `-`      | Subtraction    | Subtracts one value from another | `x - y` |
| `*`      | Multiplication | Multiplies two values            | `x * y` |
| `/`      | Division       | Divides one value by another     | `x / y` |
| `%`      | Modulus        | Returns the division remainder   | `x % y` |
| `++`     | Increment      | Increases a variable by 1        | `++x`   |
| `--`     | Decrement      | Decreases a variable by 1        | `--x`   |

## 3. Assignment Operators

Assignment operators assign values to variables.

### Example

```cpp
int x = 10;
x += 5; // x is now 15
```

```cpp
    =   +=   -=   *=   /=   %=   &=   |=   ^=   >>=   <<=
```

## 4. Comparison Operators

Comparison operators are used to compare values. They return a Boolean value: `1` (true) or `0` (false).

### Example

```cpp
int x = 5;
int y = 3;
cout << (x > y); // Outputs 1 (true)
```

| Operator | Name                     | Example  |
| -------- | ------------------------ | -------- |
| `==`     | Equal to                 | `x == y` |
| `!=`     | Not equal to             | `x != y` |
| `>`      | Greater than             | `x > y`  |
| `<`      | Less than                | `x < y`  |
| `>=`     | Greater than or equal to | `x >= y` |
| `<=`     | Less than or equal to    | `x <= y` |

## 5. Logical Operators

Logical operators test Boolean logic and are often used in conditional expressions.

```cpp
    &&   ||   !
```
