# C++: Function Overloading

## What is Function Overloading?

**Function overloading** allows multiple functions to have the same name, as long as their **parameter types** or **number of parameters** differ.

This enables you to use the same function name for similar operations on different types of data.

## Without Function Overloading

If overloading didn’t exist, you would need to use different names for similar functions:

```cpp
int plusFuncInt(int x, int y) {
  return x + y;
}

double plusFuncDouble(double x, double y) {
  return x + y;
}

int main() {
  int myNum1 = plusFuncInt(8, 5);
  double myNum2 = plusFuncDouble(4.3, 6.26);

  cout << "Int: " << myNum1 << "\n";
  cout << "Double: " << myNum2;
  return 0;
}
```

**Drawback**: Two separate function names for the same logic.

## With Function Overloading

You can define multiple functions with the same name, but different parameters:

```cpp
int plusFunc(int x, int y) {
  return x + y;
}

double plusFunc(double x, double y) {
  return x + y;
}

int main() {
  int myNum1 = plusFunc(8, 5);
  double myNum2 = plusFunc(4.3, 6.26);

  cout << "Int: " << myNum1 << "\n";
  cout << "Double: " << myNum2;
  return 0;
}
```

- Both `plusFunc` functions do the same thing.
- C++ automatically selects the correct version based on the argument types.

## Overloading by Number of Parameters

Functions can also be overloaded by **changing the number of parameters**:

```cpp
int plusFunc(int x, int y) {
  return x + y;
}

int plusFunc(int x, int y, int z) {
  return x + y + z;
}

int main() {
  int result1 = plusFunc(3, 7);       // Calls the 2-parameter version
  int result2 = plusFunc(1, 2, 3);    // Calls the 3-parameter version

  cout << "Sum of 2 numbers: " << result1 << "\n";
  cout << "Sum of 3 numbers: " << result2;
  return 0;
}
```
