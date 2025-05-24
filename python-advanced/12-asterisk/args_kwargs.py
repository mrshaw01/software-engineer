"""Variable-length arguments with *args and **kwargs"""


def my_function(*args, **kwargs):
    for arg in args:
        print(arg)
    for key in kwargs:
        print(key, kwargs[key])


my_function("hello", 42, [1, 2], name="Shaw", age=25)
