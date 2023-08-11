from typing import List, Union
from typing import TYPE_CHECKING

from EasyKnn.errors import ValueAlreadyLinkedError, ReadOnlyAttributeError, CriticalDeletionError
from EasyKnn.errors import NoDimensionError
from EasyKnn.point import Point

if TYPE_CHECKING:
    from EasyKnn.dataset import Dataset


class Value:
    """
    A Value object is the base object of the EasyKnn library. It is used to represent a set of coordinates.

    :param coordinates: The coordinates of the Value. Must be a ``list`` of ``int``, ``float`` or ``None`` values.
    :param display_name: the displayed name of the Value

    :exception NoDimensionError: If the coordinates are empty or only None values
    """

    def __init__(self, coordinates: List[Union[int, float, None]], display_name: str = None):

        if coordinates == [None] * len(coordinates):  # This way is much faster than using all()
            raise NoDimensionError("Coordinates cannot be empty or only None values")

        # We do allow the modification of the coordinates, but under certain conditions
        self._coordinates = coordinates

        # We do not allow the modification and the deletion of the dimension
        self._dimension = len(coordinates)

        # we do allow the modification and the deletion of the display_name
        self.display_name = display_name

        # We do allow the modification of the dataset, but under certain conditions
        self._dataset = None

    @property
    def coordinates(self) -> List[Union[int, float, None]]:
        """
        The coordinates of the Value.

        :read-only: False
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):

        if value == [None] * len(value):  # This way is much faster than using all()
            raise NoDimensionError("Coordinates cannot be empty or only None values")
        else:
            self._coordinates = value
            self._dimension = len(value)

    @coordinates.deleter
    def coordinates(self):
        raise CriticalDeletionError("The coordinates attribute cannot be deleted")

    # Alias for coordinates
    value = coordinates
    """
    Alias for the :attr:`coordinates<EasyKnn.value.Value.coordinates>` attribute.
    """

    @property
    def dimension(self) -> int:
        """
        The dimension of the Value. This value cannot be modified.

        :read-only: True
        """
        return self._dimension

    @dimension.setter
    def dimension(self, *args):
        raise ReadOnlyAttributeError("The dimension attribute is read-only")

    @dimension.deleter
    def dimension(self):
        raise CriticalDeletionError("The dimension attribute cannot be deleted")

    @property
    def dataset(self) -> "Dataset":
        """
        The linked :class:`Dataset<EasyKnn.dataset.Dataset>` of this Value. This value should not be modified.

        :read-only: True
        """
        return self._dataset

    @dataset.setter
    def dataset(self, *args):
        raise ReadOnlyAttributeError("The dataset attribute is read-only")

    @dataset.deleter
    def dataset(self):
        raise CriticalDeletionError("The dataset attribute cannot be deleted")

    def _set_dataset(self, value: "Dataset") -> None:
        """
        Set the dataset of the Value. This methode should only be called by the ``Dataset`` class.

        :param value: The Dataset to link the Value to
        :exception ValueAlreadyLinkedError: If the Value is already linked to a Dataset
        :return: None
        """

        if self._dataset is None:
            self._dataset = value
        else:
            raise ValueAlreadyLinkedError("This Value is already linked to a Dataset")

    def _to_point(self, distance: float) -> Point:
        """
        Convert the Value to a :class:EasyKnn.Point. This methode should only be called by the Plan class.

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
