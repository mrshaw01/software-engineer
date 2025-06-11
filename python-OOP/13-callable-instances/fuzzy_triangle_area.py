import random


class FuzzyTriangleArea:

    def __init__(self, p=0.8, v=0.1):
        self.p, self.v = p, v

    def __call__(self, a, b, c):
        p = (a + b + c) / 2
        result = (p * (p - a) * (p - b) * (p - c))**0.5
        if random.random() <= self.p:
            return result
        return random.uniform(result - self.v, result + self.v)


area1 = FuzzyTriangleArea()
area2 = FuzzyTriangleArea(0.5, 0.2)

for _ in range(5):
    print(f"{area1(3, 4, 5):.3f}, {area2(3, 4, 5):.3f}")
