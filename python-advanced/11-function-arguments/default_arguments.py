"""Default arguments in functions."""


def foo(a, b, c, d=4):
    print(a, b, c, d)


foo(1, 2, 3)
foo(1, b=2, c=3, d=100)
