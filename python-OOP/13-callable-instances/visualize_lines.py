import matplotlib.pyplot as plt
import numpy as np


class StraightLines:

    def __init__(self, m, c):
        self.slope = m
        self.y_intercept = c

    def __call__(self, x):
        return self.slope * x + self.y_intercept


lines = [StraightLines(1, 0), StraightLines(0.5, 3), StraightLines(-1.4, 1.6)]

X = np.linspace(-5, 5, 100)
for idx, line in enumerate(lines):
    Y = np.vectorize(line)(X)
    plt.plot(X, Y, label=f"line{idx}")
plt.title("Some straight lines")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
