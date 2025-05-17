from itertools import permutations

perm = permutations([1, 2, 3])
print(list(perm))

perm = permutations([1, 2, 3], 2)
print(list(perm))
