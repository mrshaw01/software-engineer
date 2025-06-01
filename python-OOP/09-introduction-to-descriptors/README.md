# Python Descriptors Tutorial

Welcome to the **Python Descriptors Tutorial**. This multi-part guide walks you through the fundamentals and advanced usage of Python’s powerful descriptor protocol—one of the core features behind `@property`, `@classmethod`, `@staticmethod`, and more.

This tutorial is organized into six clearly structured sections:

## Table of Contents

| Part                                               | Title                                      | Description                                                                                    |
| -------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| [1](./1-introduction.md)                           | **Introduction**                           | Overview of descriptors and how they integrate with Python's attribute lookup chain            |
| [2](./2-descriptor-protocol.md)                    | **Descriptor Protocol**                    | Explanation of `__get__`, `__set__`, and `__delete__`, with a simple custom descriptor example |
| [3](./3-data-non-data-descriptor.md)               | **Data vs Non-Data Descriptors**           | Difference between data and non-data descriptors, and how lookup precedence works              |
| [4](./4-practical-use-cases.md)                    | **Practical Use Cases**                    | Real-world applications: validation, caching, computed attributes, and type checking           |
| [5](./5-how-python-internally-uses-descriptors.md) | **How Python Internally Uses Descriptors** | How built-in features like `@property`, methods, and `super()` rely on descriptors             |
| [6](./6-dynamic-descriptor-creation.md)            | **Dynamic Descriptors & Best Practices**   | How to generate descriptors at runtime and best practices for maintainability                  |
