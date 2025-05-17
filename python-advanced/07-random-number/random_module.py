"""Common random operations using the `random` module."""

import random

print(random.random())  # Float [0,1)
print(random.uniform(1, 10))  # Float [1,10]
print(random.randint(1, 10))  # Int [1,10]
print(random.randrange(1, 10))  # Int [1,10)
print(random.normalvariate(0, 1))  # Normal dist float

print(random.choice("ABCDEFGHI"))  # Random element
print(random.sample("ABCDEFGHI", 3))  # k unique elements
print(random.choices("ABCDEFGHI", k=3))  # k with replacement

lst = list("ABCDEFGHI")
random.shuffle(lst)  # In-place shuffle
print(lst)
