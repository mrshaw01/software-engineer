"""Immutable dataclass with frozen=True."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImmutableRobot:
    name: str
    brandname: str


x1 = ImmutableRobot("Marvin", "NanoGuardian XR-2000")
x2 = ImmutableRobot("Marvin", "NanoGuardian XR-2000")
print(x1 == x2)
print(x1.__hash__(), x2.__hash__())
