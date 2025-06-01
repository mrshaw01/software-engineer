# 16. Dynamically Creating Classes with `type`

In this lesson, we explore how Python dynamically creates classes using the built-in `type()` function and how class definitions work behind the scenes.

## Overview

In Python, **everything is an object**, including classes themselves. This tutorial provides a deeper insight into how class creation works internally using the `type` constructor, and how it relates to metaprogramming.

Key Concepts Covered:

- `type(obj)` returns the class of an object.
- `type(class)` returns `type` because classes themselves are instances of `type`.
- Classes can be created dynamically by calling `type()` with 3 arguments.
- The usual `class` keyword is syntactic sugar for a call to `type`.

## Learning Goals

- Understand how class and `type` are related.
- Create classes dynamically using `type(name, bases, dict)`.
- See how Python processes class definitions internally.
- Compare traditional and dynamic class definitions.

## File Guide

### `01_type_basics.py`

Demonstrates how `type()` behaves on objects and classes.

### `02_class_is_instance_of_type.py`

Reveals how user-defined classes are instances of `type`.

### `03_manual_class_creation.py`

Shows how to use `type()` to create a class dynamically.

### `04_robot_vs_robot2.py`

Compares a traditional `Robot` class with a dynamically generated `Robot2` class using `type`.
