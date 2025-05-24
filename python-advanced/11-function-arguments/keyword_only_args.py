"""Forcing keyword-only arguments."""


def foo(a, b, *, c, d):
    print(a, b, c, d)


foo(1, 2, c=3, d=4)


def foo_with_args(*args, last):
    for arg in args:
        print(arg)
    print("last =", last)


foo_with_args(8, 9, 10, last=50)
