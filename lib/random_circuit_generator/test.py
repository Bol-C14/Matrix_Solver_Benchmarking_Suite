from collections import defaultdict

# Assume this is your initial list of dictionaries
dict_list = [
    {"x": 1, "y": 2, "value": 3},
    {"x": 1, "y": 2, "value": 4},
    {"x": 2, "y": 3, "value": 1},
    {"x": 2, "y": 3, "value": 1},
    {"x": 1, "y": 3, "value": 7}
]

# Create a defaultdict of int. The default value for a new key is 0
result_dict = defaultdict(float)

# Iterate over the dictionaries in the list
for d in dict_list:
    # Use tuple (x, y) as a key for the dictionary and add the value to the current sum
    result_dict[(d['x'], d['y'])] += d['value']

# Convert the result_dict back to list of dictionaries
result_list = [{"x": x, "y": y, "value": v} for (x, y), v in result_dict.items()]

# Print the result
for res in result_list:
    print(res)
