def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class SingletonClass:

    def __init__(self, data):
        self.data = data


singleton_instance_1 = SingletonClass("Instance 1")
singleton_instance_2 = SingletonClass("Instance 2")

print(singleton_instance_1 is singleton_instance_2)  # True
print(singleton_instance_1.data)  # Instance 1
print(singleton_instance_2.data)  # Instance 1
