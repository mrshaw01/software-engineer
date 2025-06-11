"""Class-based decorator that tracks how many times a function is called."""

import functools


class CountCalls:

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        print(f"Call {self.num_calls} to {self.func.__name__}")
        return self.func(*args, **kwargs)


@CountCalls
def say_hello():
    print("Hello!")


say_hello()
say_hello()
say_hello()
