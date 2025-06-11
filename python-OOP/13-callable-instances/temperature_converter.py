class TemperatureConverter:

    def __init__(self, temperature, unit="C"):
        self.temperature = temperature
        self.unit = unit

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, unit):
        if unit.upper() in {"C", "F"}:
            self.__unit = unit
        else:
            raise ValueError("Should be 'C' or 'F'")

    def convert(self):
        new_unit = "F" if self.unit == "C" else "C"
        return self._convert_to_unit(new_unit)

    def __call__(self):
        return self.temperature

    def change_unit(self, new_unit):
        new_unit = new_unit.upper()
        if new_unit != self.unit:
            self.temperature = self._convert_to_unit(new_unit)
            self.unit = new_unit

    def _convert_to_unit(self, target_unit):
        if target_unit == "C":
            return (self.temperature - 32) * 5 / 9
        return self.temperature * 9 / 5 + 32


converter = TemperatureConverter(25, "C")
print("Temp:", converter())
print("To F:", converter.convert())
converter.change_unit("F")
print("New temp:", converter())
