"""Using @dataclass(frozen=True) to create an immutable class."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImmutableRobot:
    name: str
    brandname: str


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
