"""Demonstrates deterministic results using random.seed()."""

import random


def demo(seed):
    print(f"\nSeeding with {seed}...\n")
    random.seed(seed)
    print(random.random())
    print(random.uniform(1, 10))
    print(random.choice("ABCDEFGHI"))


demo(1)
demo(42)
demo(1)
demo(42)
