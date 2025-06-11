class Robot:

    def __init__(self, name, build_year):
        self.name = name
        self.build_year = build_year

    def __repr__(self):
        return f"Robot('{self.name}', {self.build_year})"

    def __str__(self):
        return f"Name: {self.name}, Build Year: {self.build_year}"


x = Robot("Marvin", 1979)
print(repr(x))
print(str(x))
print(eval(repr(x)))
