"""Immutable class with getter-only methods (Java style)."""


class ImmutableRobot:

    def __init__(self, name, brandname):
        self.__name = name
        self.__brandname = brandname

    def get_name(self):
        return self.__name

    def get_brandname(self):
        return self.__brandname


robot = ImmutableRobot("RoboX", "TechBot")
print(robot.get_name())
print(robot.get_brandname())
