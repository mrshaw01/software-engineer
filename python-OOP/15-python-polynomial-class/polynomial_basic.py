class Polynomial:

    def __init__(self, *coefficients):
        self.coefficients = list(coefficients)

    def __repr__(self):
        return "Polynomial" + str(tuple(self.coefficients))


p = Polynomial(1, 0, -4, 3, 0)
print(p)

p2 = eval(repr(p))
print(p2)
