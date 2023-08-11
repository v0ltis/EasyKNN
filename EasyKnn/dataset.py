from typing import List

from EasyKnn.errors import ReadOnlyAttributeError
from EasyKnn.value import Value


class Dataset:
    """
    Represent a set of data
    """

    def __init__(self, display_name: str = None):
        self._data = []
        self._dataset_dimension = 0
        self.display_name = display_name

        # Only set by Neighbours class.
        # Should not be set manually.
        self._average_dist = None

    @property
    def data(self) -> List[Value]:
        """
        A list of all the values in the dataset.

        :read-only: True
        """
        return self._data

    @data.setter
    @data.deleter
    def data(self, *args):
        raise ReadOnlyAttributeError("The data cannot be modified")

    @property
    def dataset_dimension(self) -> int:
        """
        The dimension of the dataset.

        :read-only: True
        """
        return self._dataset_dimension

    @dataset_dimension.setter
    @dataset_dimension.deleter
    def dataset_dimension(self, *args):
        raise ReadOnlyAttributeError("The dataset_dimension cannot be modified")

    @property
    def average_dist(self) -> float:
        """
        The average distance of the dataset to the other datasets in the plan.

        :read-only: True
        """
        return self._average_dist

    @average_dist.setter
    @average_dist.deleter
    def average_dist(self, *args):
        raise ReadOnlyAttributeError("The average_dist cannot be modified")

    def add_value(self, value: Value):
        """
        Add a single value to the dataset

        :param value: The value to add to the dataset
        :return: None
        """
        self._data.append(value)
        self._dataset_dimension = self.get_largest_dimension()

        self.update_value_dataset()

    def add_values(self, values: List[Value]):
        """
        Add one or more values to the dataset

        :param values: A list containing the values to add to the dataset
        :return: None
        """
        self._data.extend(values)
        self._dataset_dimension = self.get_largest_dimension()

        self.update_value_dataset()

    def update_value_dataset(self) -> None:
        """
        Update the dataset attribute of all the values in the dataset

        :return: None
        """
        for value in self._data:
            if value.dataset is None:
                value._set_dataset(self)

            elif value.dataset is not self:
                raise ValueError("A single value cannot be in two different datasets")

    def get_largest_dimension(self) -> int:
        """
        Get the largest dimension of the dataset

        :return: The largest dimension of the dataset
        """
        return max([value.dimension for value in self._data])

    def nonify(self, min_dimension: int = None) -> None:
        """
        Add "None"s to all values in the dataset where the dimension is smaller than the dataset dimension.
        The min dimension is either the biggest dimension of a value in the dataset or the min_dimension parameter.

        :return: None

        >>> dataset = Dataset()
        >>> dataset.add_values([Value([1, 2, 3]), Value([4, 5]), Value([6, 7, 8, 9])])
        >>> dataset.nonify()
        >>> dataset._data
        [[1, 2, 3, None], [4, 5, None, None], [6, 7, 8, 9]]
        """

        if min_dimension is None:
            min_dimension = self._dataset_dimension

        for i in range(len(self._data)):
            coords = self._data[i]
            nonified_coords = coords.coordinates + [None] * (min_dimension - coords.dimension)
            self._data[i].coordinates = nonified_coords

    def average(self) -> List[float]:
        """
        Get the average position of all the values in the dataset. Nones values will not be counted.

        :return: A list containing the average position of all the values in the dataset

        >>> dataset = Dataset()
        >>> dataset.add_values([Value([2, 2]), Value([4, 5]), Value([6, 8])])
        >>> dataset.average()
        [4.0, 5.0]

        >>> dataset = Dataset()
        >>> dataset.add_values([Value([None, 5]), Value([None, 5, 6]), Value([None, 8]), Value([None, 2])])
        >>> dataset.nonify()
        >>> dataset.average()
        [None, 5.0, 6.0]

        >>> dataset = Dataset()
        >>> dataset.add_values([Value([1, None, None]), Value([None, 2, 2]), Value([None, -7]), Value([1, 2])])
        >>> dataset.nonify()
        >>> dataset.average()
        [1.0, -1.0, 2.0]
        """

        avg_coords = []
        for i in range(self._dataset_dimension):
            coords = [value.coordinates[i] for value in self._data if value.coordinates[i] is not None]

            if not coords:  # If coords is empty, append None, and prevent ZeroDivisionError
                avg_coords.append(None)

            else:
                avg_coords.append(sum(coords) / len(coords))

        return avg_coords

    def __repr__(self):
        return f"{self.display_name if self.display_name is not None else self._data}"

    def __str__(self):
        return f"{self.display_name if self.display_name is not None else self._data}"
