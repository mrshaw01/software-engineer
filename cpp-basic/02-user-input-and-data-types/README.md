# C++: User Input and Data Types

## 1. C++ User Input

In C++, `cin` is used to receive input from the user. It works with the extraction operator `>>` and stores input in a variable.

### Example

```cpp
int x;
cout << "Type a number: ";
cin >> x;
cout << "Your number is: " << x;
```

### Notes

- `cout` (see-out): Prints output using `<<`.
- `cin` (see-in): Gets input using `>>`.

### Simple Calculator Example

```cpp
int x, y;
int sum;
cout << "Type a number: ";
cin >> x;
cout << "Type another number: ";
cin >> y;
sum = x + y;
cout << "Sum is: " << sum;
```

## 2. C++ Data Types

Every variable in C++ must have a data type that defines what kind of value it holds.

### Common Types

```cpp
int myNum = 5;            // Integer
float myFloat = 5.99f;    // Floating point
double myDouble = 9.98;   // Double-precision float
char myChar = 'D';        // Character
bool myBool = true;       // Boolean
string myText = "Hello";  // String
```

### Data Type Summary

| Data Type | Size         | Description                           |
| --------- | ------------ | ------------------------------------- |
| `bool`    | 1 byte       | `true` or `false`                     |
| `char`    | 1 byte       | Single ASCII character                |
| `int`     | 2 or 4 bytes | Whole numbers                         |
| `float`   | 4 bytes      | Decimal numbers (6–7 digit precision) |
| `double`  | 8 bytes      | Decimal numbers (15 digit precision)  |

## 3. C++ Numeric Data Types

### Integers

```cpp
int myNum = 1000;
cout << myNum;
```

### Floating Point

```cpp
float myNum = 5.75f;
cout << myNum;
```

### Double

```cpp
double myNum = 19.99;
cout << myNum;
```

### Scientific Notation

```cpp
float f1 = 35e3;   // 35000
double d1 = 12E4;  // 120000
cout << f1;
cout << d1;
```

## 4. C++ Boolean Data Types

A boolean holds `true` or `false`, which correspond to `1` and `0` when printed.

### Example

```cpp
bool isCodingFun = true;
bool isFishTasty = false;
cout << isCodingFun;   // 1
cout << isFishTasty;   // 0
```

## 5. C++ Character Data Types

Characters are enclosed in single quotes and use ASCII codes under the hood.

### Example

```cpp
char myGrade = 'B';
cout << myGrade;
```

### ASCII Example

```cpp
char a = 65, b = 66, c = 67;
cout << a << b << c;  // Outputs: ABC
```

## 6. C++ String Data Types

Strings represent sequences of characters and require the `<string>` header.

### Example

```cpp
#include <string>
string greeting = "Hello";
cout << greeting;
```

## 7. The `auto` Keyword

Introduced in C++11, `auto` lets the compiler deduce the type of a variable.

### Example

```cpp
auto x = 5;             // int
auto myFloat = 5.99f;   // float
auto myDouble = 9.98;   // double
auto myChar = 'D';      // char
auto myBool = true;     // bool
auto myString = string("Hello");  // string
```

### Rules

- `auto` requires initialization.
- Type is fixed once assigned.

```cpp
auto x = 5;
x = 10;     // OK
x = 9.99;   // Error: x is still int
```

## 8. Real-Life Data Type Example

This program calculates the total cost of items:

### Example

```cpp
int items = 50;
double cost_per_item = 9.99;
double total_cost = items * cost_per_item;
char currency = '$';

cout << "Number of items: " << items << "\n";
cout << "Cost per item: " << cost_per_item << currency << "\n";
cout << "Total cost = " << total_cost << currency << "\n";
```
