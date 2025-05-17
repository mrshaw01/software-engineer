set_org = {1, 2, 3, 4, 5}
set_copy = set_org  # reference copy
set_copy.update([6, 7])
print("Shared ref:", set_copy)
print("Original affected:", set_org)

set_org = {1, 2, 3, 4, 5}
set_copy = set_org.copy()
set_copy.update([6, 7])
print("Copied set:", set_copy)
print("Original unchanged:", set_org)
