from typing import List, Union
from typing import TYPE_CHECKING

from EasyKnn.point import Point

if TYPE_CHECKING:
    from EasyKnn.dataset import Dataset


class Value:
    """
    A Value object is the base object of the EasyKnn library. It is used to represent a set of coordinates.

    :param coordinates: The coordinates of the Value. Must be a list of int, float or None values.
    :param display_name: the displayed name of the Value

    :exception ValueError: if the coordinates are empty or only None values
    """

    def __init__(self, coordinates: List[Union[int, float, None]], display_name: str = None):

        if coordinates == [None] * len(coordinates):  # This way is much faster than using all()
            raise ValueError("Coordinates cannot be empty or only None values")

        self.coordinates = coordinates
        self.dimension = len(coordinates)

        self.display_name = display_name

        self.dataset = None

    def to_point(self, distance: float) -> Point:
        """
        Convert the Value to a Point. This methode should only be called by the Plan class.

        :param distance: The distance between the Value and the Point
        :return: a Point object
        """

        return Point(self.coordinates, distance, self.dataset, self.display_name)

    def __repr__(self):
        return f"{self.display_name if self.display_name is not None else self.coordinates}"

    def __str__(self):
        return f"{self.display_name if self.display_name is not None else self.coordinates}"

    def __eq__(self, other):
        if not isinstance(other, Value):
            raise TypeError(f"Cannot compare Value with {type(other)}")

        return self.coordinates == other.coordinates
