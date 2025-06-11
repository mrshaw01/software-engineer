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


p = Polynomial(3, 0, -5, 2, 1)
X = np.linspace(-1.5, 1.5, 50)
F = p(X)
plt.plot(X, F)
plt.title("Polynomial Callable Example")
plt.grid()
plt.show()
