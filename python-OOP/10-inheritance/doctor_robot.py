import random


class Robot:

    def __init__(self, name):
        self.name = name
        self.health_level = random.random()

    def say_hi(self):
        print(f"Hi, I am {self.name}")

    def needs_a_doctor(self):
        return self.health_level < 0.8


class PhysicianRobot(Robot):

    def say_hi(self):
        print("Everything will be okay!")
        print(f"{self.name} takes care of you!")

    def heal(self, robo):
        robo.health_level = random.uniform(robo.health_level, 1)
        print(f"{robo.name} has been healed by {self.name}!")


doc = PhysicianRobot("Dr. Frankenstein")

for i in range(5):
    patient = Robot(f"Marvin{i}")
    if patient.needs_a_doctor():
        print(f"{patient.name} before: {patient.health_level:.2f}")
        doc.heal(patient)
        print(f"{patient.name} after: {patient.health_level:.2f}")
