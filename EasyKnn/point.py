from typing import Union, List
from typing import TYPE_CHECKING

from EasyKnn.errors import NoDimensionError, ReadOnlyAttributeError

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

    def __init__(self, coordinates: List[Union[int, float, None]],
                 distance: float, dataset: "Dataset", display_name: Union[str, None]):

        if coordinates == [None] * len(coordinates):
            raise NoDimensionError("Coordinates cannot be empty or only None values")
        else:
            self._coordinates = coordinates  # type: List[Union[int, float, None]]
            """
            Represent the coordinates of the Point.
            """
        # we do allow the modification and the deletion of the display_name
        self.display_name = display_name

        # We do not allow the modification and the deletion of the distance, and the dataset
        self._distance = distance
        self._dataset = dataset

    # We do not allow the modification and the deletion of the coordinates
    @property
    def coordinates(self) -> List[Union[int, float, None]]:
        """
        The coordinates of the Point. This value cannot be modified.

        :read-only: True
        """
        return self._coordinates

    @coordinates.setter
    @coordinates.deleter
    def coordinates(self, *args):
        raise ReadOnlyAttributeError("The coordinates attribute is read-only")

    # Alias for coordinates
    value = coordinates
    """
    This is an alias for the coordinates property.
    
    :read-only: True
    """

    # We do not allow the modification and the deletion of the distance
    @property
    def distance(self) -> float:
        """
        The distance between the Point and the Value. This value cannot be modified.

        :read-only: True
        """
        return self._distance

    @distance.setter
    @distance.deleter
    def distance(self, *args):
        raise ReadOnlyAttributeError("The distance attribute is read-only")

    @property
    def dataset(self) -> "Dataset":
        """
        The linked dataset of this Point. This value cannot be modified.

        :read-only: True
        """
        return self._dataset

    @dataset.setter
    @dataset.deleter
    def dataset(self, *args):
        raise ReadOnlyAttributeError("The dataset attribute is read-only")

    def __repr__(self):
        return f"{self.display_name if self.display_name is not None else self.coordinates}"

    def __str__(self):
        return f"{self.display_name if self.display_name is not None else self.coordinates}"
