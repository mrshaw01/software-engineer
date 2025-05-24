"""Enforce keyword-only arguments"""


def greet(name, *, age):
    print(f"Name: {name}, Age: {age}")


greet("Shaw", age=30)
# greet("Shaw", 30)  # Uncomment to raise TypeError
