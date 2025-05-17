try:
    a = 5 / 1
except ZeroDivisionError:
    print("Division failed")
else:
    print("Everything is OK")
finally:
    print("Cleaning up...")
