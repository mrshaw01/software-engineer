class Polynomial:
    def __init__(self, *coefficients):
        self.coefficients = coefficients[::-1]

    def __call__(self, x):
        return sum(coeff * x**i for i, coeff in enumerate(self.coefficients))


p1 = Polynomial(42)
p2 = Polynomial(0.75, 2)
p3 = Polynomial(1, -0.5, 0.75, 2)

for i in range(1, 10):
    print(i, p1(i), p2(i), p3(i))
