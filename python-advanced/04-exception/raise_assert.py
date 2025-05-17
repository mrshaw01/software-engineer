x = -5

try:
    if x < 0:
        raise Exception("x should not be negative.")
except Exception as e:
    print("Raised:", e)

try:
    assert x >= 0, "x is not positive."
except AssertionError as e:
    print("Assert failed:", e)
