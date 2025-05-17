"""Convert JSON strings into Python objects using json.loads()."""

import json

person_json = """
{
    "age": 30,
    "city": "New York",
    "hasChildren": false,
    "name": "John",
    "titles": ["engineer", "programmer"]
}
"""

person = json.loads(person_json)
print(person)
