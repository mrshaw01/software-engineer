"""Unpacking containers into function arguments."""


def foo(a, b, c):
    print(a, b, c)


my_list = [4, 5, 6]
foo(*my_list)

my_dict = {"a": 1, "b": 2, "c": 3}
foo(**my_dict)
