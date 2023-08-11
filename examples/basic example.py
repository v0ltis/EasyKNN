from EasyKnn import Dataset, Value, Plan


# We create two datasets

# This one is a basic one, with all values of the same length (3)
dataset = Dataset(display_name="Dataset 1")
dataset.add_values(
    [
        Value([1, 3, 3]),  # 3 values (so, it is a 3 dimensional value)
        Value([4, 5, 6])   # 3 values (so, it is a 3 dimensional value)
    ])


# This one is a bit more complex, with values of different lengths (2 and 4), with negative values and floats
dataset2 = Dataset(display_name="Dataset 2")
dataset2.add_values(
    [
        Value([6.7, 8]),        # 2 values (so, it is a 2 dimensional value)
        Value([1, 2, 7, -4.1])  # 4 values (so, it is a 4 dimensional value)
    ])

# We now create a plan, where all the given values will be placed as points
plan = Plan()

# We add the datasets to the plan
plan.add_dataset(dataset)
plan.add_dataset(dataset2)


value = Value([1, 2, 3, 5, 6])  # 5 values (so, it is a 5 dimensional value)

neighbors = plan.neighbors(value, memoize=True, nonify=True)

# Now, we can get all the data we want. Any calculation has been processed by the plan.

# We can get the nearest neighbor

nearest_neighbor = neighbors.nearest_neighbor(
    k=1,           # We want the nearest neighbor. If we want the 3 nearest neighbors, we would set k to 3

)[0]               # We add a [0] because we want the first (and only) element of the list.
                   # This function return a list of values

print(nearest_neighbor)
# >>> [[1, 2, 3, None, None]]
# Nones are appearing because the value has 5 dimensions. The nonification added missing Nones


# We now want to get the distance between the value and the nearest neighbor
print(nearest_neighbor.distance)
# >>> 1.0


# We now want to know in average, which dataset is the nearest neighbor from the value

nearest_dataset = neighbors.nearest_dataset(
    k=1,           # We want the nearest dataset. If we want the 3 nearest datasets, we would set k to 3

)[0]               # Again, we add a [0] because we want the first (and only) element of the list.

print(nearest_dataset)
# >>> Dataset 1


