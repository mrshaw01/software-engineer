"""Decode JSON back into custom objects using object_hook."""

import json


def decode_complex(dct):
    if "complex" in dct:
        return complex(dct["real"], dct["imag"])
    return dct


zJSON = '{"complex": true, "real": 5.0, "imag": 9.0}'
z = json.loads(zJSON, object_hook=decode_complex)
print(type(z))
print(z)
