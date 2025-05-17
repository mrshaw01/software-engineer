class ValueTooHighError(Exception):
    pass


class ValueTooLowError(Exception):
    def __init__(self, message, value):
        self.message = message
        self.value = value


def test_value(a):
    if a > 1000:
        raise ValueTooHighError("Value is too high.")
    if a < 5:
        raise ValueTooLowError("Value is too low.", a)
    return a


try:
    test_value(1)
except ValueTooHighError as e:
    print(e)
except ValueTooLowError as e:
    print(e.message, "The value is:", e.value)
