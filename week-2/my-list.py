# Create an empty list called my_list
my_list = []

# Append the following elements to my_list: 10, 20, 30, 40
my_list.append(10)
my_list.append(20)
my_list.append(30)
my_list.append(40)

print(my_list) # Print the current state of my_list

# Insert the value 15 at the second position (index 1) in the list
my_list.insert(1, 15)

print(my_list) # Print the current state of my_list after insertion

# Extend my_list with another list: [50, 60, 70]
my_list.extend([50, 60, 70])

print(my_list) # Print the current state of my_list after extension

# Remove the last element from my_list
my_list.pop()

print(my_list) # Print the current state of my_list after popping the last element

# Sort my_list in ascending order
my_list.sort()

print(my_list) # Print the current state of my_list after sorting

# Find and print the index of the value 30 in my_list
index_30 = my_list.index(30)
print(index_30)
