from collections import OrderedDict

ordinary_dict = {}
ordinary_dict["a"] = 1
ordinary_dict["b"] = 2
ordinary_dict["c"] = 3
ordinary_dict["d"] = 4
ordinary_dict["e"] = 5
print("Ordinary dict:", ordinary_dict)

ordered_dict = OrderedDict()
ordered_dict["a"] = 1
ordered_dict["b"] = 2
ordered_dict["c"] = 3
ordered_dict["d"] = 4
ordered_dict["e"] = 5
print("OrderedDict:", ordered_dict)

for k, v in ordered_dict.items():
    print(k, v)
