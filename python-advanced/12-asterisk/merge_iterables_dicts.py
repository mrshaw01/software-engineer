"""Merge iterables and dictionaries using unpacking"""

# Merging lists
t = (1, 2)
s = {3, 4}
print([*t, *s])

# Merging dictionaries
a = {"x": 1, "y": 2}
b = {"z": 3}
print({**a, **b})
