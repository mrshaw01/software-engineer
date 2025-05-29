# 🧬 Shallow vs Deep Copying in Python

This module demonstrates the difference between **assignment**, **shallow copy**, and **deep copy** in Python using built-in types and custom classes.

## Topics Covered

- Assignment (`obj_b = obj_a`)
- Shallow copying using `copy.copy()`
- Deep copying using `copy.deepcopy()`
- Nested lists and objects
- Copying custom objects

## 🔄 Summary Table

| Operation          | Effect                                             |
| ------------------ | -------------------------------------------------- |
| `b = a`            | Both point to the same object                      |
| `copy.copy(a)`     | One-level shallow copy — nested objects are shared |
| `copy.deepcopy(a)` | Full recursive copy — completely independent       |

## 📂 Files

| File                      | Description                                   |
| ------------------------- | --------------------------------------------- |
| `assignment_reference.py` | Demonstrates simple assignment (shared ref)   |
| `shallow_copy_flat.py`    | Shallow copy of a flat list                   |
| `shallow_copy_nested.py`  | Shallow copy of nested lists (shared inner)   |
| `deep_copy_nested.py`     | Deep copy of nested lists (full independence) |
| `custom_class_copy.py`    | Shallow/deep copy on custom classes           |
