"""Compare memory size between generator expressions and list comprehensions."""

import sys

gen_expr = (i for i in range(1000) if i % 2 == 0)
list_comp = [i for i in range(1000) if i % 2 == 0]

print("Generator expression size:", sys.getsizeof(gen_expr))
print("List comprehension size:", sys.getsizeof(list_comp))
