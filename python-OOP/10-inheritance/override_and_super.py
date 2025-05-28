class Robot:
    def __init__(self, name):
        self.name = name

    def say_hi(self):
        print(f"Hi, I am {self.name}")


class PhysicianRobot(Robot):
    def say_hi(self):
        super().say_hi()
        print("and I am a physician!")


doc = PhysicianRobot("Dr. Frankenstein")
doc.say_hi()
