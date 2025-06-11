from itertools import accumulate
import operator

print(list(accumulate([1, 2, 3, 4])))

print(list(accumulate([1, 2, 3, 4], func=operator.mul)))

print(list(accumulate([1, 5, 2, 6, 3, 4], func=max)))
