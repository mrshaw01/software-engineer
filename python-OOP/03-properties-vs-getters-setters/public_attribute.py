"""Pythonic use of public attributes without getters and setters."""


class P:

    def __init__(self, x):
        self.x = x


p1 = P(42)
p2 = P(4711)
p1.x = p1.x + p2.x
print(p1.x)
