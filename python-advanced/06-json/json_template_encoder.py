"""Generic encoder/decoder for any class with __init__ using __module__ and __class__."""

import json


class User:

    def __init__(self, name, age, active, balance, friends):
        self.name = name
        self.age = age
        self.active = active
        self.balance = balance
        self.friends = friends


class Player:

    def __init__(self, name, nickname, level):
        self.name = name
        self.nickname = nickname
        self.level = level


def encode_obj(obj):
    obj_dict = {"__class__": obj.__class__.__name__, "__module__": obj.__module__}
    obj_dict.update(obj.__dict__)
    return obj_dict


def decode_dct(dct):
    if "__class__" in dct:
        class_name = dct.pop("__class__")
        module_name = dct.pop("__module__")
        module = __import__(module_name)
        class_ = getattr(module, class_name)
        return class_(**dct)
    return dct


user = User("John", 28, True, 20.7, ["Jane", "Tom"])
user_json = json.dumps(user, default=encode_obj, indent=4)
print(user_json)
user_decoded = json.loads(user_json, object_hook=decode_dct)
print(type(user_decoded))

player = Player("Max", "max1234", 5)
player_json = json.dumps(player, default=encode_obj, indent=4)
print(player_json)
player_decoded = json.loads(player_json, object_hook=decode_dct)
print(type(player_decoded))
