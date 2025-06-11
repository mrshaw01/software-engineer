import math


class Shape:

    def __init__(self, color):
        self.color = color

    def calculate_area(self):
        pass


class Circle(Shape):

    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius

    def calculate_area(self):
        return math.pi * self.radius**2


class Rectangle(Shape):

    def __init__(self, color, width, height):
        super().__init__(color)
        self.width = width
        self.height = height

    def calculate_area(self):
        return self.width * self.height


class Triangle(Shape):

    def __init__(self, color, base, height):
        super().__init__(color)
        self.base = base
        self.height = height

    def calculate_area(self):
        return 0.5 * self.base * self.height


circle = Circle("Red", 5)
rectangle = Rectangle("Blue", 4, 6)
triangle = Triangle("Green", 3, 4)

print("Circle Area:", circle.calculate_area())
print("Rectangle Area:", rectangle.calculate_area())
print("Triangle Area:", triangle.calculate_area())
