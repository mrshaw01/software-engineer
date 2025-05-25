"""Using __slots__ to restrict attribute creation (not full immutability)."""


class ImmutableRobot:
    __slots__ = ("__name", "__brandname")

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
    robot.serial_number = 12345
except AttributeError as e:
    print(e)
