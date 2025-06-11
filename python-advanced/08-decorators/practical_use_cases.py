"""Real-world use cases for decorators: timing, debugging, validation, etc."""

import functools
import time


def timer(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result

    return wrapper


@timer
def slow_add(x, y):
    time.sleep(1)
    return x + y


print(slow_add(3, 7))
