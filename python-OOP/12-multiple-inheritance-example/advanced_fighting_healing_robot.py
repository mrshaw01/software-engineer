import random


class Robot:
    def __init__(self, name, health_level, **kwargs):
        self.name = name
        self.health_level = health_level


class HealingRobot(Robot):
    def __init__(self, healing_power, **kwargs):
        super().__init__(**kwargs)
        self.healing_power = healing_power

    def heal(self, robo):
        robo.health_level = random.uniform(robo.health_level, 1)
        print(f"{robo.name} has been healed by {self.name}!")


class FightingRobot(Robot):
    def __init__(self, fighting_power=1, **kwargs):
        super().__init__(**kwargs)
        self.fighting_power = fighting_power

    def attack(self, robo):
        robo.health_level = random.uniform(0, robo.health_level)
        print(f"{robo.name} has been attacked by {self.name}!")


class FightingHealingRobot(HealingRobot, FightingRobot):
    def __init__(self, name, health_level, healing_power, fighting_power, mode="healing", **kw):
        self.mode = mode
        super().__init__(
            name=name, health_level=health_level, healing_power=healing_power, fighting_power=fighting_power, **kw
        )

    def say_hi(self):
        if self.mode == "fighting":
            FightingRobot.say_hi(self)
        elif self.mode == "healing":
            HealingRobot.say_hi(self)
        else:
            Robot.say_hi(self)


# Run this as demo
if __name__ == "__main__":
    x = FightingHealingRobot(name="Rambo", health_level=0.9, fighting_power=0.7, healing_power=0.9)
    print(x.__dict__)
