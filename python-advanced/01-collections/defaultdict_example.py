from collections import defaultdict

d = defaultdict(int)
d["yellow"] = 1
d["blue"] = 2
print(d.items())
print("Missing key returns:", d["green"])

d = defaultdict(list)
s = [("yellow", 1), ("blue", 2), ("yellow", 3), ("blue", 4), ("red", 5)]
for k, v in s:
    d[k].append(v)

print(d.items())
print("Accessing missing list:", d["green"])
