"""Traditional class with manually defined __init__ and __repr__."""


class Robot_traditional:
    def __init__(self, model, serial_number, manufacturer):
        self.model = model
        self.serial_number = serial_number
        self.manufacturer = manufacturer

    def __repr__(self):
        return (
            f"Robot_traditional(model='{self.model}', "
            f"serial_number='{self.serial_number}', "
            f"manufacturer='{self.manufacturer}')"
        )


x = Robot_traditional("NanoGuardian XR-2000", "234-76", "Cyber Robotics Co.")
print(repr(x))
