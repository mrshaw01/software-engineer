"""Using *args and **kwargs."""


def foo(a, b, *args, **kwargs):
    print(a, b)
    for arg in args:
        print(arg)
    for key, val in kwargs.items():
        print(f"{key} = {val}")


foo(1, 2, 3, 4, 5, six=6, seven=7)
print()
foo(1, 2, three=3)
