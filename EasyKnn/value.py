from typing import List, Union


class Value:
    def __init__(self, coordinates: List[Union[int, float, None]]):

        if coordinates == [None] * len(coordinates): # This way is much faster than using all()
            raise ValueError("Coordinates cannot be empty or only None values")

        self.coordinates = coordinates
        self.dimension = len(coordinates)

    def __repr__(self):
        return f"{self.coordinates}"

    def __str__(self):
        return f"{self.coordinates}"
