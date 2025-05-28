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

    def derivative(self):
        n = len(self.coefficients)
        derived = [self.coefficients[i] * (n - i - 1) for i in range(n - 1)]
        return Polynomial(*derived)


p = Polynomial(-0.8, 2.3, 0.5, 1, 0.2)
p_der = p.derivative()

X = np.linspace(-2, 3, 100)
plt.plot(X, p(X), label="Polynomial")
plt.plot(X, p_der(X), label="Derivative")
plt.legend()
plt.grid()
plt.show()
