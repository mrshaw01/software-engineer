from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np


class Polynomial:
    def __init__(self, *coefficients):
        self.coefficients = list(coefficients)

    def __call__(self, x):
        res = 0
        for coeff in self.coefficients:
            res = res * x + coeff
        return res

    def __add__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        result = [sum(t) for t in zip_longest(c1, c2, fillvalue=0)]
        return Polynomial(*result[::-1])

    def __sub__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        result = [t1 - t2 for t1, t2 in zip_longest(c1, c2, fillvalue=0)]
        return Polynomial(*result[::-1])


p1 = Polynomial(4, 0, -4, 3, 0)
p2 = Polynomial(-0.8, 2.3, 0.5, 1, 0.2)

p_sum = p1 + p2
p_diff = p1 - p2

X = np.linspace(-3, 3, 100)
plt.plot(X, p1(X), label="p1")
plt.plot(X, p2(X), label="p2")
plt.plot(X, p_sum(X), label="p1 + p2")
plt.plot(X, p_diff(X), label="p1 - p2")
plt.legend()
plt.grid()
plt.show()
