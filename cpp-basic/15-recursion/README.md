# C++: Recursion

## What is Recursion?

**Recursion** is a programming technique where a function calls itself.
It is often used to solve complex problems by breaking them down into simpler, repeatable subproblems.

## Why Use Recursion?

- To solve problems that can be defined in terms of smaller subproblems.
- To write cleaner, more elegant solutions to tasks like tree traversal, factorials, Fibonacci numbers, etc.

## Basic Example: Sum of Numbers

Here is a simple recursive function that adds all numbers from `k` down to 0:

```cpp
int sum(int k) {
  if (k > 0) {
    return k + sum(k - 1);
  } else {
    return 0;
  }
}

int main() {
  int result = sum(10);
  cout << result;
  return 0;
}
```

### Execution Flow

When `sum(10)` is called, the function works as follows:

```
sum(10)
=> 10 + sum(9)
=> 10 + 9 + sum(8)
=> ...
=> 10 + 9 + 8 + ... + 1 + sum(0)
=> 10 + 9 + 8 + ... + 1 + 0 = 55
```

- When `k` becomes 0, the recursion stops (base case).
- Each recursive call builds up the result.

## Example: Factorial Calculation

The factorial of a number is defined as:

```
n! = n × (n - 1) × (n - 2) × ... × 1
```

Using recursion:

```cpp
int factorial(int n) {
  if (n > 1) {
    return n * factorial(n - 1);
  } else {
    return 1;
  }
}

int main() {
  cout << "Factorial of 5 is " << factorial(5);
  return 0;
}
```

### Output:

```
Factorial of 5 is 120
```

## Key Concepts in Recursion

- **Base Case**: The condition that stops the recursion.
- **Recursive Case**: The part of the function where it calls itself.

Without a proper base case, recursion may:

- Cause infinite loops
- Consume too much memory (stack overflow)

## When to Use Recursion

- Tree or graph traversal
- Divide and conquer algorithms (e.g., merge sort, quick sort)
- Problems that can be defined recursively (e.g., Fibonacci, factorial)

## Conclusion

- Recursion simplifies solving problems that have repetitive substructures.
- Always ensure a **base case** to prevent infinite recursion.
- Use recursion thoughtfully, as it can lead to performance issues if misused.
