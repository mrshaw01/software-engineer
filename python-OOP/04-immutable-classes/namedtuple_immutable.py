"""Using namedtuple from collections for immutability."""

from collections import namedtuple

ImmutableRobot = namedtuple("ImmutableRobot", ["name", "brandname"])

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
