from typing import List
from EasyKnn.value import Value


class Dataset:
    """
    Represent a set of data
    """

    def __init__(self):
        self._data = []
        self._dataset_dimension = 0

    def add_value(self, value: Value):
        self._data.append(value)
        self._dataset_dimension = self.get_largest_dimension()

        self.update_value_dataset()

    def add_values(self, values: List[Value]):
        self._data.extend(values)
        self._dataset_dimension = self.get_largest_dimension()

        self.update_value_dataset()

    def get_values(self):
        return self._data

    def update_value_dataset(self):
        for value in self._data:
            if value.dataset is None:
                value.dataset = self

            elif value.dataset is not self:
                raise ValueError("A single value cannot be in two different datasets")

    def get_largest_dimension(self):
        """
        Get the largest dimension of the dataset
        :return:
        """
        return max([value.dimension for value in self._data])

    def nonify(self, min_dimension: int = None):
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
        :return:
        >>> dataset = Dataset()
        >>> dataset.add_values([Value([2, 2]), Value([4, 5]), Value([6, 8])])
        >>> dataset.average()
        [4.0, 5.0]

        >>> dataset = Dataset()
        >>> dataset.add_values([Value([2]), Value([4, 5, 6]), Value([6, 8]), Value([1, 2])])
        >>> dataset.nonify()
        >>> dataset.average()
        [3.25, 5.0, 6.0]

        >>> dataset = Dataset()
        >>> dataset.add_values([Value([1, None, None]), Value([None, 2, 2]), Value([None, -7]), Value([1, 2])])
        >>> dataset.nonify()
        >>> dataset.average()
        [1.0, -1.0, 2.0]
        """

        avg_coords = []
        for i in range(self._dataset_dimension):
            coords = [value.coordinates[i] for value in self._data if value.coordinates[i] is not None]

            avg_coords.append(sum(coords) / len(coords))

        return avg_coords

    def __repr__(self):
        return f"{self._data}"

    def __str__(self):
        return f"{self._data}"

