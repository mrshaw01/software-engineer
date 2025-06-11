"""Manually implement an iterable class mimicking a generator."""


class FirstN:

    def __init__(self, n):
        self.n = n
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.n:
            current = self.num
            self.num += 1
            return current
        raise StopIteration()


firstn = FirstN(1_000_000)
print("Sum using custom iterable:", sum(firstn))
