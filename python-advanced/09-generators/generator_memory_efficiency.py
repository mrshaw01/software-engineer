"""Compare memory usage between list and generator for large ranges."""

import sys


# List version
def firstn_list(n):
    num, nums = 0, []
    while num < n:
        nums.append(num)
        num += 1
    return nums


print("List sum:", sum(firstn_list(1_000_000)))
print("List size:", sys.getsizeof(firstn_list(1_000_000)), "bytes")


# Generator version
def firstn_gen(n):
    num = 0
    while num < n:
        yield num
        num += 1


print("Generator sum:", sum(firstn_gen(1_000_000)))
print("Generator size:", sys.getsizeof(firstn_gen(1_000_000)), "bytes")
