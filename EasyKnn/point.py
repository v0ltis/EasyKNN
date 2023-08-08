from typing import Union, List
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from EasyKnn.dataset import Dataset


class Point:
    """
    Represent a value, after being used in a Plan.Points should only be created by the Plan class.

    Please, consider using a Value object instead.
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
