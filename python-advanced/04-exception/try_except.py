# Generic catch
try:
    a = 5 / 0
except:
    print("Some error occurred.")

# Catch specific exception
try:
    a = 5 / 0
except ZeroDivisionError as e:
    print("Handled ZeroDivisionError:", e)

# Multiple errors
try:
    a = 5 / 1
    b = a + "10"
except ZeroDivisionError as e:
    print("ZeroDivisionError:", e)
except TypeError as e:
    print("TypeError:", e)
