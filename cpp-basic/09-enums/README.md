# C++ Enums

## Introduction

An **enum** (short for _enumeration_) is a user-defined type in C++ that consists of a set of named integral constants. Enums are commonly used to represent a group of related constants with more readable names.

## Declaring an Enum

Use the `enum` keyword followed by the name of the enum, then list the constant values:

```cpp
enum Level {
  LOW,
  MEDIUM,
  HIGH
};
```

- Values are separated by commas.
- The last item does **not** require a trailing comma.
- The first item is assigned `0` by default, the second `1`, and so on.

## Creating and Assigning Enum Variables

To use an enum, declare a variable of the enum type and assign it a value:

```cpp
enum Level myVar;
myVar = MEDIUM;
```

You can also initialize it directly:

```cpp
enum Level myVar = MEDIUM;
```

Printing the value will show its underlying integer:

```cpp
cout << myVar;  // Outputs 1
```

## Customizing Enum Values

You can assign specific integer values to enum items:

```cpp
enum Level {
  LOW = 25,
  MEDIUM = 50,
  HIGH = 75
};
```

You can mix manual and automatic assignments:

```cpp
enum Level {
  LOW = 5,
  MEDIUM,  // becomes 6
  HIGH     // becomes 7
};
```

## Using Enums in a Switch Statement

Enums are often used in switch statements to simplify branching logic:

```cpp
enum Level {
  LOW = 1,
  MEDIUM,
  HIGH
};

int main() {
  enum Level myVar = MEDIUM;

  switch (myVar) {
    case 1:
      cout << "Low Level";
      break;
    case 2:
      cout << "Medium level";
      break;
    case 3:
      cout << "High level";
      break;
  }

  return 0;
}
```

## When to Use Enums

Enums are useful when:

- You want to group related constants under a common type.
- You want to improve code readability and reduce the risk of using magic numbers.
- You are working with fixed categories like status codes, days, states, directions, etc.
