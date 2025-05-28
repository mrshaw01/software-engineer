x = 42
try:
    x.a = "not possible"
except AttributeError as e:
    print("Error when adding attribute to int:", e)

lst = [34, 999, 1001]
try:
    lst.a = "forget it"
except AttributeError as e:
    print("Error when adding attribute to list:", e)
