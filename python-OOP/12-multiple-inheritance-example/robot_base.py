import random


class Robot:
    __illegal_names = {"Henry", "Oscar"}
    __crucial_health_level = 0.6

    def __init__(self, name):
        self.name = name
        self.health_level = random.random()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = "Marvin" if name in Robot.__illegal_names else name

    def __str__(self):
        return self.name + ", Robot"

    def __add__(self, other):
        first = self.name.split("-")[0]
        second = other.name.split("-")[0]
        return type(self)(f"{first}-{second}")

    def needs_a_nurse(self):
        return self.health_level < Robot.__crucial_health_level

    def say_hi(self):
        print(f"Hi, I am {self.name}")
        print(f"My health level is: {self.health_level}")
