# C++: Math and Boolean Logic

## C++ Math

C++ provides a variety of functions to perform mathematical operations.

### Max and Min

- `max(x, y)` returns the larger of the two values.
- `min(x, y)` returns the smaller of the two values.

```cpp
cout << max(5, 10);  // Outputs 10
cout << min(5, 10);  // Outputs 5
```

### `<cmath>` Library

For more complex math operations, include the `<cmath>` header:

```cpp
#include <cmath>

cout << sqrt(64);    // Square root
cout << round(2.6);  // Rounds to 3
cout << log(2);      // Natural log of 2
```

> 📘 For more math functions, refer to the C++ Math Reference.

## C++ Booleans

Booleans are used to store values that are either `true` or `false`.

```cpp
bool isCodingFun = true;
bool isFishTasty = false;

cout << isCodingFun;   // Outputs 1 (true)
cout << isFishTasty;   // Outputs 0 (false)
```

## C++ Boolean Expressions

A Boolean expression evaluates to `true` (1) or `false` (0):

```cpp
int x = 10, y = 9;
cout << (x > y);        // Outputs 1
cout << (10 == 15);     // Outputs 0
```

## Real-World Boolean Example

```cpp
int myAge = 25;
int votingAge = 18;

cout << (myAge >= votingAge);  // Outputs 1 (true)
```

Wrap it with an `if...else` for a better experience:

```cpp
if (myAge >= votingAge) {
  cout << "Old enough to vote!";
} else {
  cout << "Not old enough to vote.";
}
```

## C++ Conditions and If Statements

C++ supports the following logical conditions:

- `a < b`, `a <= b`, `a > b`, `a >= b`
- `a == b`, `a != b`

### The `if` Statement

```cpp
if (20 > 18) {
  cout << "20 is greater than 18";
}
```

### The `else` Statement

```cpp
int time = 20;

if (time < 18) {
  cout << "Good day.";
} else {
  cout << "Good evening.";
}
```

### The `else if` Statement

```cpp
int time = 22;

if (time < 10) {
  cout << "Good morning.";
} else if (time < 20) {
  cout << "Good day.";
} else {
  cout << "Good evening.";
}
```

## Short Hand If...Else (Ternary Operator)

```cpp
int time = 20;
string result = (time < 18) ? "Good day." : "Good evening.";
cout << result;
```

## More Boolean Examples

### Door Lock

```cpp
int doorCode = 1337;

if (doorCode == 1337) {
  cout << "Correct code.\nThe door is now open.\n";
} else {
  cout << "Wrong code.\nThe door remains closed.\n";
}
```

### Positive or Negative

```cpp
int myNum = 10;

if (myNum > 0) {
  cout << "The value is a positive number.\n";
} else if (myNum < 0) {
  cout << "The value is a negative number.\n";
} else {
  cout << "The value is 0.\n";
}
```

### Even or Odd

```cpp
int myNum = 5;

if (myNum % 2 == 0) {
  cout << myNum << " is even.\n";
} else {
  cout << myNum << " is odd.\n";
}
```

## C++ Switch Statements

The `switch` statement allows you to execute different blocks of code based on the value of a variable.

### Basic Syntax

```cpp
switch(expression) {
  case x:
    // code
    break;
  case y:
    // code
    break;
  default:
    // code
}
```

### Example

```cpp
int day = 4;

switch (day) {
  case 1:
    cout << "Monday"; break;
  case 2:
    cout << "Tuesday"; break;
  case 3:
    cout << "Wednesday"; break;
  case 4:
    cout << "Thursday"; break;
  case 5:
    cout << "Friday"; break;
  case 6:
    cout << "Saturday"; break;
  case 7:
    cout << "Sunday"; break;
}
// Outputs: Thursday
```

### The `break` Keyword

Used to exit the switch block once a match is found. If omitted, execution will continue to the next case.

### The `default` Keyword

Specifies code to run when no case matches:

```cpp
int day = 4;

switch (day) {
  case 6:
    cout << "Today is Saturday"; break;
  case 7:
    cout << "Today is Sunday"; break;
  default:
    cout << "Looking forward to the Weekend";
}
// Outputs: Looking forward to the Weekend
```
