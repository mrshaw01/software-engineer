class Robot:

    def __init__(self, name):
        self.name = name
        print(f"{self.name} has been created!")

    def __del__(self):
        print(f"{self.name} is being destroyed")


x = Robot("Tik-Tok")
y = Robot("Jenkins")
z = x
del x
del z
del y
