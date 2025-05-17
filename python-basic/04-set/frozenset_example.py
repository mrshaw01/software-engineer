a = frozenset([0, 1, 2, 3, 4])
# a.add(5)  # Error: immutable

odds = frozenset({1, 3, 5, 7, 9})
evens = frozenset({0, 2, 4, 6, 8})

print("Union:", odds.union(evens))
print("Intersection:", odds.intersection(evens))
print("Difference:", odds.difference(evens))
