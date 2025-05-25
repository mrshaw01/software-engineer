"""
Ccy class for handling currency values and conversion.
Supports +, +=, * operations, including scalar and mixed-type expressions.
"""


class Ccy:
    currencies = {"CHF": 1.08, "CAD": 1.49, "GBP": 0.89, "JPY": 114.39, "EUR": 1.0, "USD": 1.11}

    def __init__(self, value, unit="EUR"):
        self.value = value
        self.unit = unit

    def __str__(self):
        return f"{self.value:.2f} {self.unit}"

    def __repr__(self):
        return f'Ccy({self.value}, "{self.unit}")'

    def changeTo(self, new_unit):
        self.value = self.value / Ccy.currencies[self.unit] * Ccy.currencies[new_unit]
        self.unit = new_unit

    def __add__(self, other):
        if isinstance(other, (int, float)):
            x = other * Ccy.currencies[self.unit]
        else:
            x = other.value / Ccy.currencies[other.unit] * Ccy.currencies[self.unit]
        return Ccy(self.value + x, self.unit)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            x = other * Ccy.currencies[self.unit]
        else:
            x = other.value / Ccy.currencies[other.unit] * Ccy.currencies[self.unit]
        self.value += x
        return self

    def __radd__(self, other):
        result = self + other
        if result.unit != "EUR":
            result.changeTo("EUR")
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Ccy(self.value * other, self.unit)
        raise TypeError(f"unsupported operand type(s) for *: 'Ccy' and '{type(other).__name__}'")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.value *= other
            return self
        raise TypeError(f"unsupported operand type(s) for *=: 'Ccy' and '{type(other).__name__}'")


if __name__ == "__main__":
    x = Ccy(10, "USD")
    y = Ccy(11)
    z = Ccy(12.34, "JPY")
    z = 7.8 + x + y + 255 + z
    print(z)
    lst = [Ccy(10, "USD"), Ccy(11), Ccy(12.34, "JPY"), Ccy(12.34, "CAD")]
    z = sum(lst)
    print(z)
