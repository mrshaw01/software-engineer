class Animal:
    def __init__(self, name, age, sound):
        self.name = name
        self.age = age
        self.sound = sound

    def make_sound(self):
        print(f"{self.name} says: {self.sound}")


class Mammal(Animal):
    def __init__(self, name, age, sound, fur_color, number_of_legs):
        super().__init__(name, age, sound)
        self.fur_color = fur_color
        self.number_of_legs = number_of_legs

    def give_birth(self, name):
        return Mammal(name, 0, self.sound, self.fur_color, self.number_of_legs)

    def nurse_young(self):
        print(f"{self.name} nurses its young.")


class Bird(Animal):
    def __init__(self, name, age, sound, wingspan):
        super().__init__(name, age, sound)
        self.wingspan = wingspan

    def fly(self):
        print(f"{self.name} flies with a wingspan of {self.wingspan}.")


class Reptile(Animal):
    def __init__(self, name, age, sound, scale_color):
        super().__init__(name, age, sound)
        self.scale_color = scale_color

    def crawl(self):
        print(f"{self.name} crawls with {self.scale_color} scales.")


dog = Mammal("Molly", 5, "Woof", "Brown", 4)
eagle = Bird("Eagle", 3, "Screech", "Large")
turtle = Reptile("Turtle", 10, "Hiss", "Green")

dog.make_sound()
dog.give_birth("Charlie").make_sound()
eagle.fly()
turtle.crawl()
