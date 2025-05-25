"""
Length class with support for unit conversion and operator overloading.
Supports addition, in-place addition, and reverse addition.
"""


class Length:
    __metric = {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}

    def __init__(self, value, unit="m"):
        self.value = value
        self.unit = unit

    def Converse2Metres(self):
        return self.value * Length.__metric[self.unit]

    def __add__(self, other):
        if isinstance(other, (int, float)):
            l = self.Converse2Metres() + other
        else:
            l = self.Converse2Metres() + other.Converse2Metres()
        return Length(l / Length.__metric[self.unit], self.unit)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            l = self.Converse2Metres() + other
        else:
            l = self.Converse2Metres() + other.Converse2Metres()
        self.value = l / Length.__metric[self.unit]
        return self

    def __str__(self):
        return str(round(self.Converse2Metres(), 5))

    def __repr__(self):
        return f"Length({self.value}, '{self.unit}')"


if __name__ == "__main__":
    x = Length(4)
    print(x)
    y = eval(repr(x))
    z = Length(4.5, "yd") + Length(1)
    z += Length(2, "m")
    print(repr(z))
    print(z)
