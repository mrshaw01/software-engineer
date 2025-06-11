"""Immutable class using @property with no setters."""


class ImmutableRobot:

    def __init__(self, name, brandname):
        self.__name = name
        self.__brandname = brandname

    @property
    def name(self):
        return self.__name

    @property
    def brandname(self):
        return self.__brandname


robot = ImmutableRobot("RoboX", "TechBot")
print(robot.name)
print(robot.brandname)

try:
    robot.name = "RoboY"
except AttributeError as e:
    print(e)

try:
    robot.brandname = "NewTechBot"
except AttributeError as e:
    print(e)
