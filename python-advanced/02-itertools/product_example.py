from itertools import product

# Cartesian product
prod = product([1, 2], [3, 4])
print(list(prod))

# Repeating product
prod = product([1, 2], [3], repeat=2)
print(list(prod))
