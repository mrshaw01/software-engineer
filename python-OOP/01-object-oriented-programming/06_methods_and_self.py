class Robot:
    def __init__(self, name):
        self.name = name

    def say_hi(self):
        print(f"Hi, I am {self.name}")


x = Robot("Marvin")
x.say_hi()
