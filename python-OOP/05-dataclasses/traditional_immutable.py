"""Manual implementation of immutable class with __eq__ and __hash__."""


class ImmutableRobot_traditional:

    def __init__(self, name: str, brandname: str):
        self._name = name
        self._brandname = brandname

    @property
    def name(self) -> str:
        return self._name

    @property
    def brandname(self) -> str:
        return self._brandname

    def __eq__(self, other):
        if not isinstance(other, ImmutableRobot_traditional):
            return False
        return self.name == other.name and self.brandname == other.brandname

    def __hash__(self):
        return hash((self.name, self.brandname))


x1 = ImmutableRobot_traditional("Marvin", "NanoGuardian XR-2000")
x2 = ImmutableRobot_traditional("Marvin", "NanoGuardian XR-2000")
print(x1 == x2)
print(x1.__hash__(), x2.__hash__())
