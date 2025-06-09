# C++ Loops

Loops are used to repeatedly execute a block of code as long as a given condition is true. They help reduce repetition, make code more concise, and minimize errors.

## 🔁 While Loop

The `while` loop runs a block of code **as long as** a condition remains true.

### Syntax

```cpp
while (condition) {
  // code block to be executed
}
```

### Example: Count up

```cpp
int i = 0;
while (i < 5) {
  cout << i << "\n";
  i++;
}
```

> ⚠️ Don't forget to update the variable inside the loop to avoid infinite loops.

### Example: Countdown

```cpp
int countdown = 3;
while (countdown > 0) {
  cout << countdown << "\n";
  countdown--;
}
cout << "Happy New Year!!\n";
```

## 🔄 Do/While Loop

The `do/while` loop runs **at least once**, then continues if the condition is true.

### Syntax

```cpp
do {
  // code block to be executed
} while (condition);
```

### Example

```cpp
int i = 0;
do {
  cout << i << "\n";
  i++;
} while (i < 5);
```

### Even When Condition is False

```cpp
int i = 10;
do {
  cout << "i is " << i << "\n";
  i++;
} while (i < 5);
```

### Practical Example: User Input

```cpp
int number;
do {
  cout << "Enter a positive number: ";
  cin >> number;
} while (number > 0);
```

## 🔁 For Loop

Use a `for` loop when you know in advance how many times you want to execute a block of code.

### Syntax

```cpp
for (initialization; condition; increment) {
  // code block
}
```

### Example: Print 0 to 4

```cpp
for (int i = 0; i < 5; i++) {
  cout << i << "\n";
}
```

### Other Examples

- **Even numbers:**

  ```cpp
  for (int i = 0; i <= 10; i += 2) {
    cout << i << "\n";
  }
  ```

- **Sum from 1 to 5:**

  ```cpp
  int sum = 0;
  for (int i = 1; i <= 5; i++) {
    sum += i;
  }
  cout << "Sum is " << sum;
  ```

- **Countdown:**

  ```cpp
  for (int i = 5; i > 0; i--) {
    cout << i << "\n";
  }
  ```

## 🔁 Nested Loops

A loop inside another loop.

### Example: Outer and Inner Loops

```cpp
for (int i = 1; i <= 2; ++i) {
  cout << "Outer: " << i << "\n";
  for (int j = 1; j <= 3; ++j) {
    cout << " Inner: " << j << "\n";
  }
}
```

### Multiplication Table

```cpp
for (int i = 1; i <= 3; i++) {
  for (int j = 1; j <= 3; j++) {
    cout << i * j << " ";
  }
  cout << "\n";
}
```

## 🔁 For-Each Loop (Range-Based For)

Available since C++11. Use it to iterate over arrays or collections.

### Syntax

```cpp
for (type var : array) {
  // code block
}
```

### Example: Array

```cpp
int myNumbers[5] = {10, 20, 30, 40, 50};
for (int i : myNumbers) {
  cout << i << "\n";
}
```

### Example: String Characters

```cpp
string word = "Hello";
for (char c : word) {
  cout << c << "\n";
}
```

## 🔚 Break and Continue

### Break: Exit loop early

```cpp
for (int i = 0; i < 10; i++) {
  if (i == 4) break;
  cout << i << "\n";
}
```

### Continue: Skip current iteration

```cpp
for (int i = 0; i < 10; i++) {
  if (i == 4) continue;
  cout << i << "\n";
}
```

### In While Loops

**Break:**

```cpp
int i = 0;
while (i < 10) {
  cout << i << "\n";
  i++;
  if (i == 4) break;
}
```

**Continue:**

```cpp
int i = 0;
while (i < 10) {
  if (i == 4) {
    i++;
    continue;
  }
  cout << i << "\n";
  i++;
}
```
