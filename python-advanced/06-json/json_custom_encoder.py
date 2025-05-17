"""Encode custom Python objects (e.g. complex numbers) with a function or JSONEncoder."""

import json
from json import JSONEncoder


def encode_complex(z):
    if isinstance(z, complex):
        return {"complex": True, "real": z.real, "imag": z.imag}
    raise TypeError(f"Not serializable: {type(z)}")


class ComplexEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return {"complex": True, "real": o.real, "imag": o.imag}
        return super().default(o)


z = 5 + 9j
try:
    print(json.dumps(z))
except TypeError as e:
    print(f"Error: {e}")
print(json.dumps(z, default=encode_complex))
print(json.dumps(z, cls=ComplexEncoder))
