"""Yield Fibonacci numbers lazily with a generator."""


def fibonacci(limit):
    a, b = 0, 1
    while b < limit:
        yield b
        a, b = b, a + b


print("Fibonacci < 30:", list(fibonacci(30)))
