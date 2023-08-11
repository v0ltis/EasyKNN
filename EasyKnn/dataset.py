from typing import List

from EasyKnn.errors import ReadOnlyAttributeError, ValueAlreadyLinkedError, CriticalDeletionError, \
    DatasetAlreadyLinkedError
from EasyKnn.value import Value


class Dataset:
    """
    Represent a collection of :class:`Values<EasyKnn.value.Value>`. This class will be then used to create
    a :class:`Plan<EasyKnn.plan.Plan>`.

    :param display_name: The displayed name of the Dataset
    """

    def __init__(self, display_name: str = None):
        self._data = []
        self._dataset_dimension = 0
        self.display_name = display_name
        self._liked_plan = None

        # Only set by Neighbors class.
        # Should not be set manually.
        self._average_dist = None

    @property
    def data(self) -> List[Value]:
        """
        A list of all the :class:`Value<EasyKnn.value.Value>` in the Dataset.

        :read-only: True
        """
        return self._data

    @data.setter
    def data(self, *args):
        raise ReadOnlyAttributeError("The data attribute is read-only")

    @data.deleter
    def data(self, *args):
        raise CriticalDeletionError("The data attribute cannot be deleted")

    @property
    def dataset_dimension(self) -> int:
        """
        The dimension of the Dataset. A dimension is the largest number of coordinates a
        :class:`Value<EasyKnn.value.Value>` has in the whole Dataset.

        :read-only: True
        """
        return self._dataset_dimension

    @dataset_dimension.setter
    def dataset_dimension(self, *args):
        raise ReadOnlyAttributeError("The dataset_dimension attribute is read-only")

    @dataset_dimension.deleter
    def dataset_dimension(self, *args):
        raise ReadOnlyAttributeError("The dataset_dimension attribute cannot be modified")

    @property
    def average_dist(self) -> float:
        """
        The average distance of the Dataset to the :class:`Value<EasyKnn.value.Value>`. This attribute is ``None``
        until :meth:`Plan.neighbors<EasyKnn.plan.Plan.neighbors>` is called.

        :read-only: True
        """
        return self._average_dist

    @average_dist.setter
    def average_dist(self, *args):
        raise ReadOnlyAttributeError("The average_dist attribute is read-only")

    @average_dist.deleter
    def average_dist(self, *args):
        raise CriticalDeletionError("The average_dist attribute cannot be deleted")

    @property
    def linked_plan(self):
        """
        The :class:`Plan<EasyKnn.plan.Plan>` the Dataset is linked to. Editing this attribute is strongly discouraged.

        :read-only: False
        """
        return self._liked_plan

    @linked_plan.setter
    def linked_plan(self, plan):
        if self._liked_plan is None:
            self._liked_plan = plan
        else:
            raise DatasetAlreadyLinkedError("This dataset is already linked to a plan")

    @linked_plan.deleter
    def linked_plan(self):
        raise CriticalDeletionError("The linked_plan attribute cannot be deleted")

    def add_value(self, value: Value):
        """
        Add a single :class:`Value<EasyKnn.value.Value>` to the Dataset

        :param value: The :class:`Value<EasyKnn.value.Value>` to add to the Dataset
        :return: ``None``
        """
        self._data.append(value)
        self._dataset_dimension = self.get_largest_dimension()

        self._update_value_dataset()

    def add_values(self, values: List[Value]):
        """
        Add one or more :class:`Values<EasyKnn.value.Value>` to the Dataset

        :param values: A list containing the :class:`Values<EasyKnn.value.Value>` to add to the Dataset
        :return: ``None``
        """
        self._data.extend(values)
        self._dataset_dimension = self.get_largest_dimension()

        self._update_value_dataset()

    def _update_value_dataset(self) -> None:
        """
        Update the :atr:`Dataset<EasyKnn.value.Value.dataset>` attribute of all the
        :class:`Values<EasyKnn.value.Value>` in the Dataset

        :return: ``None``
        """
        for value in self._data:
            if value.dataset is None:
                value._set_dataset(self)

            elif value.dataset is not self:
                raise ValueAlreadyLinkedError("A single value cannot be in two different datasets")

    def get_largest_dimension(self) -> int:
        """
        Get the largest dimension of the Dataset. The largest dimension of all the :class:`Values<EasyKnn.value.Value>`

        :return: The largest dimension of the Dataset
        """
        return max([value.dimension for value in self._data])

    def nonify(self, min_dimension: int = None) -> None:
        """
        Add ``None`` to all :class:`Values<EasyKnn.value.Value>` in the Dataset where the dimension is smaller than the
        Dataset dimension.

        :param min_dimension: The minimum length of the :class:`Values<EasyKnn.value.Value>` in the Dataset.
                If ``None``, the Dataset dimension will be used.
        :return: ``None``

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
        Get the average position of all the values in the Dataset. Nones values will not be counted.

        :return: A list containing the average position of all the values in the Dataset

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
