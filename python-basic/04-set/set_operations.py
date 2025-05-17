odds = {1, 3, 5, 7, 9}
evens = {0, 2, 4, 6, 8}
primes = {2, 3, 5, 7}

print("Union:", odds.union(evens))
print("Intersection:", odds.intersection(primes))
print("Evens ∩ Primes:", evens.intersection(primes))

setA = {1, 2, 3, 4, 5, 6, 7, 8, 9}
setB = {1, 2, 3, 10, 11, 12}

print("A - B:", setA.difference(setB))
print("B - A:", setB.difference(setA))
print("A △ B:", setA.symmetric_difference(setB))
