# In this example, we will create use the Weight functionality.
# We will try to find the best car for a customer based on several criteria.

from EasyKnn import Value, Dataset, Weight, Plan

# First, we will create a dataset with the cars we have in our dealership.

dealership = Dataset()


# We now add cars to the dealership. Each car has 4 values: [Fuel consumption, Engine power, Top speed, Price]
dealership.add_values(
    [
        Value([7.5, 130, 210, 30000], "Mercedes C-Class"),
        Value([8.1, 120, 195, 25000], "Peugeot 508"),
        Value([6.8, 160, 235, 40000], "BMW 3-Series"),
        Value([7.3, 115, 185, 22000], "Skoda Octavia"),
        Value([9.2, 100, 170, 20000], "Toyota Prius Sedan"),
        Value([4.2, 45, 45, 7000], "Citroen Ami"),
    ])

# We will now create a customer that wants a car with the following criteria:
# Fuel consumption: 7.5
# Engine power: 150
# Top speed: 220
# Price: 30000

Bob = Value([7.5, 180, 220, 35000])

# Bob will now look for a car in the dealership that matches his criteria.
# He wants a car with a high engine power and top speed, but don't really care about the price.

# We will now create a weight for each value in the dataset.

bob_weight = Weight([1, 2, 2, 0.4])

# We will now find the best car for Bob.
# First, we will add the dataset to a plan.
plan = Plan()

plan.add_dataset(dealership)

# Then, we will calculate the best car for Bob.
best_car = plan.neighbors(Bob, weight=bob_weight)

print("The best car for Bob is:", best_car.neighbors[0])
# The best car for Bob is: BMW 3-Series

# Now, someone who do not care at all the fuel consumption of the car she is looking for

alice = Value([7.5, 125, 180, 23000])  # The fuel consumption could also be 0, since it weight will be 0.

alice_weight = Weight([0, 1, 1, 0.7])

best_car = plan.neighbors(alice, weight=alice_weight)

print("The best car for Alice is:", best_car.neighbors[0])
# The best car for Alice is: Škoda Octavia

# Now, someone who just want the cheapest car

charlie = Value([7.5, 125, 180, 0])

charlie_weight = Weight([0, 0, 0, 1])  # We set the wanted price to 0, since it is the only value that matters.
                                       # The nearest car will be the cheapest one.

best_car = plan.neighbors(charlie, weight=charlie_weight, use_abs=False)

print("The best car for Charlie is:", best_car.neighbors[0])
# The best car for Charlie is: Citroen Ami

