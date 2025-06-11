class Robot:

    def __init__(self, name=None):
        self.name = name

    def say_hi(self):
        if self.name:
            print(f"Hi, I am {self.name}")
        else:
            print("Hi, I am a robot without a name")


x = Robot()
y = Robot("Marvin")
x.say_hi()
y.say_hi()
