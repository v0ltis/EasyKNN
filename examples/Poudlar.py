# In this example, we will use the Harry Potter lore.
# We will try to find the house of a student, based of the scores of other students.
# We will also try to find his potential friends.

# In this example, only 4 arbitrary values are used. In a real case, you would use more values
# We will use the following values:
# [Courage, Intelligence, Strength, Logic]

# First, we need to import the library
from EasyKnn import Plan, Dataset, Value

# We will create all the Values and Dataset we need

Gryffindor = Dataset(display_name="Gryffindor")  # The dataset of the Gryffindor house

Gryffindor.add_values(
    [
        Value([10, 5, 8, 3], display_name="Harry Potter"),    # Here, Harry potter has 10 in courage, 5 in
                                                                        # intelligence, 8 in strength and 3 in logic.
        Value([9, 6, 7, 4], display_name="Hermione Granger"),
        Value([2, 7, 6, 5], display_name="Ron Weasley"),
        Value([7, 8, 5, 6], display_name="Neville Longbottom"),
    ]
)

Ravenclaw = Dataset(display_name="Ravenclaw")  # The dataset of the Ravenclaw house

Ravenclaw.add_values(
    [
        Value([5, 9, 3, 8], display_name="Luna Lovegood"),
        Value([6, 9, 4, 7], display_name="Cho Chang"),
        Value([7, 8, 5, 5], display_name="Padma Patil"),
        Value([7, 7, 6, 5], display_name="Terry Boot"),
    ]
)

Hufflepuff = Dataset(display_name="Hufflepuff")  # The dataset of the Hufflepuff house

Hufflepuff.add_values(
    [
        Value([3, 8, 5, 10], display_name="Cedric Diggory"),
        Value([4, 7, 6, 9], display_name="Hannah Abbott"),
        Value([5, 6, 7, 8], display_name="Susan Bones"),
        Value([6, 5, 8, 7], display_name="Justin Finch-Fletchley"),
    ]
)

Slytherin = Dataset(display_name="Slytherin")  # The dataset of the Slytherin house

Slytherin.add_values(
    [
        Value([7, 3, 10, 1], display_name="Draco Malfoy"),
        Value([7, 4, 9, 6], display_name="Vincent Crabbe"),
        Value([6, 5, 8, 7], display_name="Gregory Goyle"),
        Value([5, 6, 7, 7], display_name="Pansy Parkinson"),

    ]

)

# Congratulations, you have created all the datasets you need !

# Now, we will create the plan

plan = Plan()

# We will add all the datasets to the plan

plan.add_datasets([Gryffindor, Ravenclaw, Hufflepuff, Slytherin])

# Since all the values are 4-dimensional, there is no need to nonify anything.


# Now, we will create the value of the student we want to find the house

student = Value([8, 4, 2, 7], display_name="Hughes Pham")  # Here, Hughes Pham has 8 in courage, 4 in intelligence,
                                                                      # 2 in strength and 7 in logic.

# Now, we will calculate the distance between the student and all the other values

neighbors = plan.neighbors(student, memoize=True, nonify=False)


# Now, we will display 3 student who will be the most likely to be friends with Hughes Pham
likely_friends = neighbors.nearest_neighbor(3)

for i in range(3):
    print(
        f"{likely_friends[i].display_name} of house {likely_friends[i].dataset} is {likely_friends[i].distance} away from {student.display_name}"
    )

# This will print Neville Longbottom, Terry Boot and Cho Chang

print("\n\n\n")

# We also want to know the 3 students who will be the least likely to be friends with Hughes Pham
unlikely_friends = neighbors.nearest_neighbor(-3)  # A negative number will return the furthest neighbors

for i in range(3):
    print(
        f"{unlikely_friends[i].display_name} of house {unlikely_friends[i].dataset} is {unlikely_friends[i].distance} away from {student.display_name}"
    )


# This will print Draco Malfoy, Ron Weasley, Cedric Diggory.

print("\n\n\n")

# Now, we will find the house of Hughes Pham !

house = neighbors.nearest_dataset(
    1,  # We only want to find the nearest dataset

)[0]  # We only want the element of the list, not the list itself

print(f"{student.display_name} will probably join {house.display_name} !")
# Hughes will probably join Ravenclaw !

# The interesting part, is that the most likely friends of Hughes is not from Ravenclaw, but from Gryffindor !


# We also can know the average distance between Hughes and the students of each house
for i in range(4):
    print(f"The average distance between {student.display_name} and the students of {plan.datasets[i].display_name} is {plan.datasets[i].average_dist}")

# We find that Hughes is in fact, waayy more likely to be in Ravenclaw than in Gryffindor !
# Also, he is very unlikely to be in Slytherin, since the average distance is very high (~ 7.5 with Slytherin
# when only ~ 5.7 with Ravenclaw)
