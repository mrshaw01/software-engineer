"""Unpack iterable and dictionary into function arguments"""


def foo(a, b, c):
    print(a, b, c)


foo(*[1, 2, 3])
foo(*"XYZ")
foo(**{"a": 10, "b": 20, "c": 30})
