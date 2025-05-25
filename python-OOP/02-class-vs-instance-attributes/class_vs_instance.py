"""Demonstrates how class and instance attributes behave differently."""


class A:
    a = "I am a class attribute!"


x = A()
y = A()

print("x.a:", x.a)
print("y.a:", y.a)
print("A.a:", A.a)

x.a = "This creates a new instance attribute for x!"
print("After x.a change:")
print("x.a:", x.a)
print("y.a:", y.a)
print("A.a:", A.a)

A.a = "This is changing the class attribute 'a'!"
print("After A.a change:")
print("x.a:", x.a)
print("y.a:", y.a)
print("A.a:", A.a)

print("x.__dict__:", x.__dict__)
print("y.__dict__:", y.__dict__)
print("A.__dict__:", A.__dict__)
