import sys
import timeit

my_list = [0, 1, 2, "hello", True]
my_tuple = (0, 1, 2, "hello", True)

print(sys.getsizeof(my_list), "bytes")
print(sys.getsizeof(my_tuple), "bytes")

print(timeit.timeit(stmt="[0, 1, 2, 3, 4, 5]", number=1_000_000))
print(timeit.timeit(stmt="(0, 1, 2, 3, 4, 5)", number=1_000_000))

# 104 bytes  ← list
#  80 bytes  ← tuple

# 0.03388... seconds ← list creation 1 million times
# 0.00469... seconds ← tuple creation 1 million times

# timeit.timeit() measures the execution time of creating the list vs. tuple 1 million times.

# Tuples are faster to create because:
# - There is no dynamic memory allocation needed.
# - Internally, tuples are simpler data structures.
