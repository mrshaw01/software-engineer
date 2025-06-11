"""Decorator that preserves function metadata using functools.wraps."""

import functools


def start_end_decorator(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Start")
        result = func(*args, **kwargs)
        print("End")
        return result

    return wrapper


@start_end_decorator
def add_5(x):
    """Adds 5 to the input."""
    return x + 5


print(add_5(10))
print(add_5.__name__)
print(add_5.__doc__)
