"""Decorator factory that accepts arguments (e.g., repeat a function multiple times)."""

import functools


def repeat(n):

    def decorator_repeat(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator_repeat


@repeat(3)
def greet(name):
    print(f"Hello, {name}!")


greet("Alex")
