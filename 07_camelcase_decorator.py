import re


# Class decorator to convert camelCase method names to snake_case
def camel_case_to_underscore(cls):
    modified_items = {}

    for key, value in cls.__dict__.items():
        if callable(value) and re.match(r"^[a-z]+(?:[A-Z][a-z]*)+$", key):
            # Convert camelCase to snake_case
            underscore_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
            modified_items[underscore_name] = value
        else:
            modified_items[key] = value

    return type(cls.__name__, cls.__bases__, modified_items)


@camel_case_to_underscore
class CamelCaseClass:
    def processData(self):
        print("Processing data...")

    def transformData(self):
        print("Transforming data...")

    def processOutputData(self):
        print("Processing output data...")


# Creating an instance and calling snake_case methods
camel_case_instance = CamelCaseClass()
camel_case_instance.process_data()
camel_case_instance.transform_data()
camel_case_instance.process_output_data()
