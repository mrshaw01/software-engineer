"""Decorator that supports arbitrary arguments and preserves return value."""


def start_end_decorator(func):

    def wrapper(*args, **kwargs):
        print("Start")
        result = func(*args, **kwargs)
        print("End")
        return result

    return wrapper


@start_end_decorator
def add_5(x):
    return x + 5


print(add_5(10))
