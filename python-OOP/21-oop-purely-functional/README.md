# 21. OOP Purely Functional

This chapter bridges Object-Oriented Programming (OOP) and Functional Programming (FP) by demonstrating how functional techniques can emulate OOP behavior such as encapsulation, state management, and interface methods.

## Concepts Covered

- Encapsulation using closures
- Local state via `nonlocal` variables
- Getter and Setter design patterns in functional style
- Comparison between functional and class-based OOP

## Functional Robot

The `robot_functional.py` file defines a `Robot` function that uses closures to mimic private variables and methods in OOP.

## Class-Based Robot

The `robot_class.py` file shows a traditional class-based approach for comparison.

## Comparison

Both styles provide:

- Controlled access to internal state
- Independent instances with encapsulated data
- Method-like interfaces

## Run Examples

```bash
python robot_functional.py
python robot_class.py
```
