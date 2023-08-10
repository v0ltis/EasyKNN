from typing import Union, List
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from EasyKnn.dataset import Dataset


class Point:
    """
    A point object is an extension of the Value object. It is used after the Plan `neighbors` method.
    This object should not be created by the user, but only by the Plan class.

    :param coordinates: The coordinates of the Point. Must be a list of int, float or None values.
    :param distance: the distance between the Point and the Value
    :param dataset: the linked dataset of this Point
    :param display_name: the displayed name of the Point

    :exception NoDimensionError: if the coordinates are empty or only None values
    """

    def __init__(self, coordinates: List[Union[str, float, None]],
                 distance: float, dataset: "Dataset", display_name: Union[str, None]):

        self.coordinates = coordinates
        self.display_name = display_name
        self.distance = distance
        self.dataset = dataset

    def __repr__(self):
        return f"{self.display_name if self.display_name is not None else self.coordinates}"

    def __str__(self):
        return f"{self.display_name if self.display_name is not None else self.coordinates}"
