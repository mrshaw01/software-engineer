"""Generic property behavior using __getattr__ and __setattr__."""


class Robot:

    def __init__(self, name, build_year, city):
        self.name = name
        self.build_year = build_year
        self.city = city

    def __getattr__(self, name):
        return self.__dict__[f"__{name}"]

    def __setattr__(self, name, value):
        self.__dict__[f"__{name}"] = value


robot = Robot("RoboBot", 2022, "TechCity")
print(robot.name)
print(robot.build_year)
print(robot.city)
print(robot.__dict__)
