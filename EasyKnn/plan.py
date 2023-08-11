from typing import List

from EasyKnn.errors import ReadOnlyAttributeError, CriticalDeletionError
from EasyKnn.neighbours import Neighbours
from EasyKnn.value import Value
from EasyKnn.dataset import Dataset
from EasyKnn.weight import Weight


class Plan:
    """
    Represents a plan for a KNN algorithm.
    There will be represented each value in an X-dimensional space.
    """
    def __init__(self):
        self._datasets = []
        self._memoized = {}

    @property
    def datasets(self) -> List[Dataset]:
        """
        A list of all the datasets of the plan

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
        A dictionary of all the memoized values of the plan

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
        Add a single dataset to the plan

        :param dataset: The dataset to add to the plan
        :return: None

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
        Add one or more datasets to the plan
        :param datasets: A list of datasets to add to the plan
        :return: None

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
        Clear the cache of the plan
        :return:
        """
        self._memoized = {}

    def _distance(self, value: Value, point: Value, weights: Weight, memoize: bool = True, use_abs = True) -> float:
        """
        Get the distance between two values
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

    def neighbours(self, value: Value, memoize: bool = True,
                   nonify: bool = True, weight: Weight = Weight([]),
                   use_abs: bool = True) -> Neighbours:
        """
        Get the k nearest neighbors of a value

        :param value: The value to get the neighbors
        :param memoize: If you want to memoize the distances between the values. This will make the algorithm faster,
                        but will use more memory.
        :param nonify: If all dataset should be nonized to the same dimension as the given value.
        :param weight: The weight to use for the distance calculation. By default, each dimension will have a weight
                        set to 1.
        :param use_abs: If the absolute value of the distance should be used. Useless with the default weight,
                        but if negatives weights are used, this can be usefull. Disabling this will make the algorithm
                        considering a distance of -7 nearest than 0 for example. Enabling this will make the algorithm
                        considering a distance of 0 nearest than -7.
        :return: A Neighbours object
        """

        points = []

        if nonify:
            for dataset in self.datasets:
                dataset.nonify(value.dimension)

        values = [value for dataset in self.datasets for value in dataset.data]

        for point in values:
            distance = self._distance(value, point, weight, memoize, use_abs=use_abs)

            points.append(point.to_point(distance))

        return Neighbours(points)
