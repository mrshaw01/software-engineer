"""Demonstrate failure when merging dicts with non-string keys using dict() constructor"""

a = {"one": 1, "two": 2}
b = {3: "three", "four": 4}

# Raises: TypeError: keywords must be strings
# print(dict(a, **b))

# Correct way
print({**a, **b})
