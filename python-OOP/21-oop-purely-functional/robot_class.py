class Robot:
    def __init__(self, name, city):
        self._name = name
        self._city = city

    def get_name(self):
        return self._name

    def set_name(self, new_name):
        self._name = new_name

    def get_city(self):
        return self._city

    def set_city(self, new_city):
        self._city = new_city


# Create robot objects
marvin = Robot("Marvin", "New York")
r2d2 = Robot("R2D2", "Tatooine")

print(marvin.get_name(), marvin.get_city())
marvin.set_name("Marvin 2.0")
print(marvin.get_name(), marvin.get_city())

print(r2d2.get_name(), r2d2.get_city())
r2d2.set_city("Naboo")
print(r2d2.get_name(), r2d2.get_city())
