"""Using frozen dataclass in sets and dictionaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImmutableRobot:
    name: str
    brandname: str


robot1 = ImmutableRobot("Marvin", "NanoGuardian XR-2000")
robot2 = ImmutableRobot("R2D2", "QuantumTech Sentinel-7")
robot3 = ImmutableRobot("Marva", "MachinaMaster MM-42")

robots = {robot1, robot2, robot3}

print("The robots in the set robots:")
for robo in robots:
    print(robo)

activity = {robot1: "activated", robot2: "activated", robot3: "deactivated"}

print("\nAll the activated robots:")
for robo, mode in activity.items():
    if mode == "activated":
        print(f"{robo} is activated")
