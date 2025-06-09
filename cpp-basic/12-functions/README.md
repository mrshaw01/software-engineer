# C++: Functions

## What is a Function?

A **function** is a block of code that runs only when it is called. Functions can take **parameters** and can optionally **return values**.

Functions help with code reusability—write once, use many times.

## Creating a Function

Syntax:

```cpp
void myFunction() {
  // code to be executed
}
```

- `void` means the function does not return a value.
- The body contains the code to execute when the function is called.

## Calling a Function

Declared functions are not run until you call them:

```cpp
int main() {
  myFunction();  // function call
  return 0;
}
```

You can call the same function multiple times.

## Function Declaration and Definition

A function has two parts:

- **Declaration** (function prototype)
- **Definition** (function body)

If you define a function **after `main()`**, you must declare it before:

```cpp
// Declaration
void myFunction();

int main() {
  myFunction();  // OK
  return 0;
}

// Definition
void myFunction() {
  cout << "I just got executed!";
}
```

## Function Parameters and Arguments

You can pass data to functions using **parameters**:

```cpp
void greet(string name) {
  cout << "Hello " << name << endl;
}
```

Calling:

```cpp
greet("Alice");  // "Alice" is an argument
```

## Default Parameters

You can give parameters default values:

```cpp
void greet(string name = "Guest") {
  cout << "Hello " << name << endl;
}

greet();            // Hello Guest
greet("Charlie");   // Hello Charlie
```

## Multiple Parameters

You can pass multiple parameters:

```cpp
void introduce(string name, int age) {
  cout << name << " is " << age << " years old.\n";
}
```

Make sure to match the number and order of arguments when calling.

## Returning Values

Use `return` to return values from a function:

```cpp
int add(int a, int b) {
  return a + b;
}

int main() {
  int result = add(3, 5);
  cout << result;  // 8
}
```

## Practical Example

```cpp
int doubleValue(int x) {
  return x * 2;
}

int main() {
  for (int i = 1; i <= 5; i++) {
    cout << "Double of " << i << " is " << doubleValue(i) << endl;
  }
}
```

## Pass by Reference

Use **references** to modify original values:

```cpp
void swap(int &a, int &b) {
  int temp = a;
  a = b;
  b = temp;
}
```

You can also modify strings by reference:

```cpp
void appendWorld(string &str) {
  str += " World!";
}
```

## C++ Pass Array to a Function

You can also pass **arrays** to functions as parameters.

### Example

```cpp
void myFunction(int myNumbers[5]) {
  for (int i = 0; i < 5; i++) {
    cout << myNumbers[i] << "\n";
  }
}

int main() {
  int myNumbers[5] = {10, 20, 30, 40, 50};
  myFunction(myNumbers);
  return 0;
}
```

### Explanation

- The function `myFunction` accepts an array as a parameter: `int myNumbers[5]`.
- Inside the function, a `for` loop prints each element.
- When calling the function, you pass only the **name** of the array (e.g., `myFunction(myNumbers)`).
- Note: The size (`[5]`) in the parameter is optional—it is ignored by the compiler.

## C++ Pass Structures to a Function

You can pass **structures** to functions to work with grouped data.

### Pass by Value

```cpp
struct Car {
  string brand;
  int year;
};

void printCarInfo(Car c) {
  cout << "Brand: " << c.brand << ", Year: " << c.year << "\n";
}

int main() {
  Car myCar = {"Toyota", 2020};
  printCarInfo(myCar);
  return 0;
}
```

- This passes a **copy** of the `Car` structure.
- Changes made inside the function will **not** affect the original structure.

### Pass by Reference

To modify the original structure or avoid copying large data, pass it by reference using `&`:

```cpp
struct Car {
  string brand;
  int year;
};

void updateYear(Car& c) {
  c.year++;
}

int main() {
  Car myCar = {"Toyota", 2020};
  updateYear(myCar);
  cout << "The " << myCar.brand << " is now from year " << myCar.year << ".\n";
  return 0;
}
```

- Now, the original structure is modified inside the function.
- Use reference when:

  - You want to **change** the original structure.
  - You want to **avoid copying** large structures.
