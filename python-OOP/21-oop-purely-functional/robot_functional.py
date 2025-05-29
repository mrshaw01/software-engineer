def Robot(name, city):
    def self():
        return None

    def get_name():
        return name

    def set_name(new_name):
        nonlocal name
        name = new_name

    def get_city():
        return city

    def set_city(new_city):
        nonlocal city
        city = new_city

    self.get_name = get_name
    self.set_name = set_name
    self.get_city = get_city
    self.set_city = set_city

    return self


# Create robot objects
marvin = Robot("Marvin", "New York")
r2d2 = Robot("R2D2", "Tatooine")

print(marvin.get_name(), marvin.get_city())
marvin.set_name("Marvin 2.0")
print(marvin.get_name(), marvin.get_city())

print(r2d2.get_name(), r2d2.get_city())
r2d2.set_city("Naboo")
print(r2d2.get_name(), r2d2.get_city())
