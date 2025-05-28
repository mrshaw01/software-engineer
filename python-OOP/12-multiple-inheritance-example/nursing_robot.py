import random

from robot_base import Robot


class NursingRobot(Robot):
    def __init__(self, name="Hubert", healing_power=None):
        super().__init__(name)
        self.healing_power = healing_power or random.uniform(0.8, 1)

    def say_hi(self):
        print(f"Well, well, everything will be fine ... {self.name} takes care of you!")

    def say_hi_to_doc(self):
        Robot.say_hi(self)

    def heal(self, robo):
        if robo.health_level > self.healing_power:
            print(f"{self.name} not strong enough to heal {robo.name}")
        else:
            robo.health_level = random.uniform(robo.health_level, self.healing_power)
            print(f"{robo.name} has been healed by {self.name}!")
