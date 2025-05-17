from itertools import combinations, combinations_with_replacement

comb = combinations([1, 2, 3, 4], 2)
print(list(comb))

comb_wr = combinations_with_replacement([1, 2, 3, 4], 2)
print(list(comb_wr))
