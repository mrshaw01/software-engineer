# C++ Strings

## Introduction

Strings in C++ are used to store sequences of characters such as words or sentences.

```cpp
#include <string>

string greeting = "Hello";
```

To use strings, include the `<string>` header.

## String Concatenation

Strings can be concatenated using the `+` operator or the `append()` method:

```cpp
string firstName = "John";
string lastName = "Doe";
string fullName = firstName + " " + lastName;
```

```cpp
string fullName = firstName.append(lastName);
```

## Numbers and Strings

C++ uses `+` for both addition and string concatenation.

```cpp
int x = 10, y = 20;
int z = x + y;  // 30

string a = "10", b = "20";
string c = a + b;  // "1020"
```

Mixing string and number directly will cause an error:

```cpp
string a = "10";
int b = 20;
// string c = a + b;  // Error!
```

## String Length

Use `.length()` or `.size()` to get the number of characters in a string:

```cpp
string txt = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
cout << txt.length();
cout << txt.size();  // Same result
```

## Accessing Characters

Access string characters via index or the `.at()` method:

```cpp
string myString = "Hello";
cout << myString[0];        // H
cout << myString.at(1);     // e
cout << myString[myString.length() - 1]; // o

myString[0] = 'J';           // Change to "Jello"
myString.at(0) = 'J';        // Same effect
```

## Special Characters

Escape sequences allow special characters in strings:

| Escape | Meaning      |
| ------ | ------------ |
| `\'`   | Single quote |
| `\"`   | Double quote |
| `\\`   | Backslash    |
| `\n`   | New line     |
| `\t`   | Tab          |

Example:

```cpp
string txt = "We are the so-called \"Vikings\" from the north.";
```

## User Input Strings

Basic input using `cin` only reads a single word:

```cpp
string name;
cin >> name;
```

To read full lines, use `getline()`:

```cpp
string fullName;
getline(cin, fullName);
```

## Namespace Usage

You can either:

- Use `using namespace std;`, or
- Prefix with `std::`:

```cpp
std::string greeting = "Hello";
std::cout << greeting;
```

## C-Style Strings

C-style strings use `char` arrays:

```cpp
char greeting[] = "Hello";  // C-style string
```

These are less convenient than `std::string` but still used for compatibility with C.
