import re


class CamelCaseToUnderscoreMeta(type):
    def __new__(cls, name, bases, dct):
        modified_items = {}
        for key, value in dct.items():
            if callable(value) and re.match(r"^[a-z]+(?:[A-Z][a-z]*)*$", key):
                underscore_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
                modified_items[underscore_name] = value
            else:
                modified_items[key] = value
        return super().__new__(cls, name, bases, modified_items)


class CamelCaseClass(metaclass=CamelCaseToUnderscoreMeta):
    def processData(self):
        print("Processing data...")

    def transformData(self):
        print("Transforming data...")

    def processOutputData(self):
        print("Processing output data...")


camel_case_instance = CamelCaseClass()
camel_case_instance.process_data()
camel_case_instance.transform_data()
camel_case_instance.process_output_data()
