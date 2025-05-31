"""Real-world classmethod example to count people."""


class Person:
    total_people = 0

    def __init__(self, name):
        self.name = name
        Person.total_people += 1

    def __del__(self):
        Person.total_people -= 1
        print(f"{self.name} has been removed from the count.")

    @classmethod
    def display_total_people(cls):
        print("Total number of people:", cls.total_people)


person1 = Person("Alice")
person2 = Person("Bob")
person3 = Person("Charlie")
del person2  # Deleting an instance does not affect the class attribute

Person.display_total_people()
