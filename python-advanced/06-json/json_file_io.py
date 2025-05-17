"""Read/write JSON to/from files using json.dump and json.load."""

import json
import os

person = {"name": "John", "age": 30, "city": "New York", "hasChildren": False, "titles": ["engineer", "programmer"]}

# Ensure the directory exists
os.makedirs("sample_files", exist_ok=True)

with open("sample_files/person.json", "w") as f:
    json.dump(person, f, indent=2)

with open("sample_files/person.json", "r") as f:
    data = json.load(f)
    print(data)
