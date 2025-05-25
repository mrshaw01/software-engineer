"""Basic usage of @dataclass."""

from dataclasses import dataclass


@dataclass
class Robot:
    model: str
    serial_number: str
    manufacturer: str


y = Robot("MachinaMaster MM-42", "986-42", "Quantum Automations Inc.")
print(repr(y))
