from typing import List

from EasyKnn.errors import ReadOnlyAttributeError, CriticalDeletionError
from EasyKnn.neighbors import Neighbors
from EasyKnn.value import Value
from EasyKnn.dataset import Dataset
from EasyKnn.weight import Weight


class Plan:
    """
    In a Plan, all :class:`Values<EasyKnn.value.Value` will be represented in an X-dimensional space as a point.
    """
    def __init__(self):
        self._datasets = []
        self._memoized = {}

    @property
    def datasets(self) -> List[Dataset]:
        """
        A list of all the :class:`Datasets`<EasyKnn.datasets.Datasets>` of the Plan

        :read-only: True
        """
        return self._datasets

    @datasets.setter
    def datasets(self, *args):
        raise ReadOnlyAttributeError("The datasets attribute is read-only")

    @datasets.deleter
    def datasets(self, *args):
        raise CriticalDeletionError("The datasets attribute cannot be deleted")

    @property
    def memoized(self) -> dict:
        """
        A dictionary containing all the memoized distances of the Plan during the execution of
        the :meth:`neighbors<EasyKnn.plan.Plan.neighbors>` method.

        :read-only: True
        """
        return self._memoized

    @memoized.setter
    def memoized(self, *args):
        raise ReadOnlyAttributeError("The memoized attribute is read-only")

    @memoized.deleter
    def memoized(self, *args):
        raise CriticalDeletionError("The memoized attribute cannot be deleted")

    def add_dataset(self, dataset: Dataset):
        """
        Add a single :class:`Dataset<EasyKnn.dataset.Dataset>` to the Plan

        :param dataset: The :class:`Dataset<EasyKnn.dataset.Dataset>` to add to the plan
        :return: ``None``

        >>> plan = Plan()
        >>> dataset = Dataset()
        >>> dataset.add_values([Value([1, 2, 3]), Value([4, 5, 6])])
        >>> plan.add_dataset(dataset)
        >>> plan.datasets
        [[[1, 2, 3], [4, 5, 6]]]
        """

        # We will add the plan to the dataset.
        # An error will be raised if the dataset is already linked to a plan
        dataset.linked_plan = self

        self._datasets.append(dataset)

    def add_datasets(self, datasets: List[Dataset]):
        """
        Add one or more :class:`Datasets<EasyKnn.dataset.Dataset>` to the Plan

        :param datasets: A list of :class:`Datasets<EasyKnn.dataset.Dataset>` to add to the Plan
        :return: ``None``

        >>> plan = Plan()
        >>> dataset1 = Dataset()
        >>> dataset1.add_values([Value([1, 2, 3]), Value([4, 5, 6])])
        >>> dataset2 = Dataset()
        >>> dataset2.add_values([Value([7, 8, 9]), Value([10, 11, 12])])
        >>> plan.add_datasets([dataset1, dataset2])
        >>> plan.datasets
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        """

        # We will add the plan to each dataset.
        # An error will be raised if any dataset is already linked to a plan

        for dataset in datasets:
            dataset.linked_plan = self

        self._datasets.extend(datasets)

    def clear_cache(self):
        """
        Clear the :attr:`Cache<EasyKnn.plan.Plan.memoized>` of the Plan's
        :attr:`memoized distances<EasyKnn.plan.Plan.memoized>`.

        :return: ``None``
        """
        self._memoized = {}

    def _distance(self, value: Value, point: Value, weights: Weight, memoize: bool = True, use_abs=True) -> float:
        """
        Get the distance between two :class:`Values<EasyKnn.value.Value>`.

        :param value: The first :class:`Value<EasyKnn.value.Value>` to compare. When used by the
                :meth:`neighbors<EasyKnn.plan.Plan.neighbors>` method, this is the main
                :class:`Value<EasyKnn.value.Value>, compared to all the :class:`Values<EasyKnn.value.Value>`
                of the added datasets.
        :param point: The second :class:`Value<EasyKnn.value.Value>` to compare. When used by the
                :meth:`neighbors<EasyKnn.plan.Plan.neighbors>`method, this will be a point from the Plan object.
        :param weights: The :class:`Weight<EasyKnn.weight.Weight>` to use to calculate the distance. By default, an
                empty :class:`Weight<EasyKnn.weight.Weight>` will be used, which means that all the dimensions will
                have the same weight.
        :param memoize: If True, the distance will be memoized in the :attr:`Cache<EasyKnn.plan.Plan.memoized>`
                of the plan.
        :param use_abs: If True, the absolute value of the distance will be returned. If False, a negative value could
                be returned if the Weight is negative.

        :return: The distance between the two values

        >>> value = Value([1, 2, 2])
        >>> point = Value([2, 2, 2])

        >>> plan = Plan()
        >>> plan._distance(value, point, Weight([1, 1, 1]))
        1.0

        >>> value = Value([None, 2, 2])
        >>> point = Value([2, 2, 2])
        >>> plan._distance(value, point, Weight([1, 1, 1]))
        0.0
        """

        coord_sum = 0

        tupled = (tuple(value.coordinates), tuple(point.coordinates))

        if tupled in self._memoized and memoize is True:
            # We will use the memoized value if it is available
            return self._memoized[tupled]

        else:
            for i in range(len(value.coordinates)):
                value_coord = value.coordinates[i]
                point_coord = point.coordinates[i]

                if value_coord is None or point_coord is None:
                    # We will ignore this step, since the value is not defined in this dimension.
                    pass

                else:
                    # We will add the square of the difference between the two coordinates
                    coord_sum += (value_coord - point_coord) ** 2 * weights[i]  # Euclidean distance, multiplied by the weight.
                                                                                # If weight[i] is not defined, 1 will nbe returned

            # we take the square root of the sum of the squares.
            # this work for any number of dimensions
            result = coord_sum ** 0.5

            # If a weight or a value is negative, the distance will be complex. We will return a non-complex value.
            if isinstance(result, complex):
                # We will return the absolute value of the distance.

                # Despite the use_abs parameter, we will always use the absolute value of the distance.
                # In fact, use_abs determines if the distance should always be positive or not.
                # A negative distance is considered as nearest than 0 by the algorithm,
                # even if it's not mathematically true.
                result = abs(result) * -1 if not use_abs else abs(result)

            if memoize is True:
                self._memoized[tupled] = result

            return result

    def neighbors(self, value: Value, memoize: bool = True,
                  nonify: bool = True, weight: Weight = Weight([]),
                  use_abs: bool = True) -> Neighbors:
        """
        Get the k nearest neighbors of a value

        :param value: The :class:`Value<EasyKnn.value.Value>` to get the neighbors
        :param memoize: If you want to memoize the distances between the :class:`Value<EasyKnn.value.Value>`.
                    This will make the algorithm faster,but will use more memory.
        :param nonify: If all dataset should be :meth:`nonified<EasyKnn.dataset.Dataset.nonify>` to the same dimension
                    as the given :class:`Value<EasyKnn.value.Value>`.
        :param weight: The :class:`Weight<EasyKnn.weight.Weight>` to use for the distance calculation. By default, each
                    dimension will have a weight set to 1.
        :param use_abs: If the absolute value of the distance should be used. Useless with the default weight,
                    but if negatives weights are used, this can be useful. Disabling this will make the algorithm
                    considering a distance of -7 nearest than 0 for example. Enabling it will make the algorithm
                    considering a distance of -7 equal to 7, and so further than 0.
        :return: A :class:`Neighbors<EasyKnn.neighbors.Neighbors>` object containing the nearest neighbors and datasets
        """

        points = []

        if nonify:
            for dataset in self.datasets:
                dataset.nonify(value.dimension)

        values = [value for dataset in self.datasets for value in dataset.data]

        for point in values:
            distance = self._distance(value, point, weight, memoize, use_abs=use_abs)

            points.append(point._to_point(distance))

        return Neighbors(points)
