"""Use numpy.random for scientific-grade random generation."""

import numpy as np

np.random.seed(1)
print(np.random.rand(3))  # Array of 3 floats

np.random.seed(1)
print(np.random.rand(3))  # Reproducible

print(np.random.randint(0, 10, (5, 3)))  # 5x3 int array

print(np.random.randn(5))  # Standard normal

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.random.shuffle(arr)  # Shuffle rows
print(arr)
