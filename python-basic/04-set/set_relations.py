setA = {1, 2, 3, 4, 5, 6}
setB = {1, 2, 3}
setC = {7, 8, 9}

print("B ⊆ A:", setB.issubset(setA))
print("A ⊇ B:", setA.issuperset(setB))
print("A ⊥ C:", setA.isdisjoint(setC))
