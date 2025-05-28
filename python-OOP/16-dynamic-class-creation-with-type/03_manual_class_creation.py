# Creating class A dynamically
A = type("A", (), {})
x = A()

print("Type of x:", type(x))  # <class '__main__.A'>
print("Class name:", x.__class__.__name__)
