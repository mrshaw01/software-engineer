"""Basic decorator example that adds behavior before and after a function."""


def start_end_decorator(func):
    def wrapper():
        print("Start")
        func()
        print("End")

    return wrapper


@start_end_decorator
def greet():
    print("Hello!")


greet()
