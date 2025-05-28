import random

from robot_base import Robot


class FightingRobot(Robot):
    __maximum_damage = 0.2

    def __init__(self, name="Hubert", fighting_power=None):
        super().__init__(name)
        max_dam = FightingRobot.__maximum_damage
        self.fighting_power = fighting_power or random.uniform(max_dam, 1)

    def say_hi(self):
        print(f"I am the terrible ... {self.name}")

    def attack(self, other):
        other.health_level *= self.fighting_power
        if isinstance(other, FightingRobot):
            self.health_level *= other.fighting_power
