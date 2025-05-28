class TriangleArea:
    def __call__(self, a, b, c):
        p = (a + b + c) / 2
        return (p * (p - a) * (p - b) * (p - c)) ** 0.5


area = TriangleArea()
print(area(3, 4, 5))  # Output: 6.0
