"""Example of using multiple decorators (stacked execution order)."""

import functools


def debug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result

    return wrapper


def start_end(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Start")
        result = func(*args, **kwargs)
        print("End")
        return result

    return wrapper


@debug
@start_end
def say_hello(name):
    print(f"Hello, {name}!")
    return f"Greeted {name}"


say_hello("Alex")
