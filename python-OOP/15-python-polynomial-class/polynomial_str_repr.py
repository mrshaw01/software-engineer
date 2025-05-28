class Polynomial:
    def __init__(self, *coefficients):
        self.coefficients = list(coefficients)

    def __repr__(self):
        return "Polynomial" + str(tuple(self.coefficients))

    def __str__(self):
        def x_expr(degree):
            return "" if degree == 0 else "x" if degree == 1 else f"x^{degree}"

        degree = len(self.coefficients) - 1
        res = ""
        for i, coeff in enumerate(self.coefficients):
            if coeff == 0:
                continue
            sign = "+" if coeff > 0 else "-"
            coeff_abs = abs(coeff)
            coeff_str = "" if coeff_abs == 1 and i != degree else str(coeff_abs)
            term = f"{sign}{coeff_str}{x_expr(degree - i)}"
            res += term

        return res.lstrip("+")


polys = [
    Polynomial(1, 0, -4, 3, 0),
    Polynomial(2, 0),
    Polynomial(4, 1, -1),
    Polynomial(3, 0, -5, 2, 7),
    Polynomial(-42),
]

for i, poly in enumerate(polys):
    print(f"$p_{i} = {str(poly)}$")
