class Robot:

    def __init__(self, name):
        self.name = name

    def say_hi(self):
        print(f"Hi, I am {self.name}")


class PhysicianRobot(Robot):
    pass


x = Robot("Marvin")
y = PhysicianRobot("James")

x.say_hi()
y.say_hi()
