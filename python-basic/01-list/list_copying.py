# Reference copy (both point to the same object)
list_org = ["banana", "cherry", "apple"]
list_copy = list_org
list_copy.append(True)
print("Shared ref:", list_copy)
print("Original list also changed:", list_org)

# Actual copy
list_org = ["banana", "cherry", "apple"]
list_copy = list_org.copy()
list_copy.append(False)
print("Copied list:", list_copy)
print("Original unchanged:", list_org)
