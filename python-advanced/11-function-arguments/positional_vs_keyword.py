"""Positional and keyword arguments."""


def foo(a, b, c):
    print(a, b, c)


foo(1, 2, 3)
foo(a=1, b=2, c=3)
foo(c=3, b=2, a=1)
foo(1, b=2, c=3)
