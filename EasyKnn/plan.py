from typing import List

from EasyKnn.neighbours import Neighbours
from EasyKnn.value import Value
from EasyKnn.dataset import Dataset


class Plan:
    """
    Represents a plan for a KNN algorithm.
    There will be represented each value in an X-dimensional space.
    """
    def __init__(self):
        self.datasets = []
        self.memoized = {}

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
        self.datasets.append(dataset)

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
        self.datasets.extend(datasets)

    def clear_cache(self):
        """
        Clear the cache of the plan
        :return:
        """
        self.memoized = {}

    def _distance(self, value: Value, point: Value, memoize: bool = True) -> float:
        """
        Get the distance between two values
        :return: The distance between the two values

        >>> value = Value([1, 2, 2])
        >>> point = Value([2, 2, 2])

        >>> plan = Plan()
        >>> plan._distance(value, point)
        1.0

        >>> value = Value([None, 2, 2])
        >>> point = Value([2, 2, 2])
        >>> plan._distance(value, point)
        0.0
        """

        coord_sum = 0

        tupled = (tuple(value.coordinates), tuple(point.coordinates))

        if tupled in self.memoized and memoize is True:
            # We will use the memoized value if it is available
            return self.memoized[tupled]

        else:
            for i in range(len(value.coordinates)):
                value_coord = value.coordinates[i]
                point_coord = point.coordinates[i]

                if value_coord is None or point_coord is None:
                    # We will ignore this step, since the value is not defined in this dimension.
                    pass

                else:
                    # We will add the square of the difference between the two coordinates
                    coord_sum += (value_coord - point_coord) ** 2

            # we take the square root of the sum of the squares.
            # this work for any number of dimensions
            result = coord_sum ** 0.5

            if memoize is True:
                self.memoized[tupled] = result

            return result

    def neighbours(self, value: Value, memoize: bool = True, nonify: bool = True) -> Neighbours:
        """
        Get the k nearest neighbors of a value

        :param value: The value to get the neighbors
        :param output: Either if you want the neither dataset of the neither value from the given value.
        :param memoize: If you want to memoize the distances between the values. This will make the algorithm faster, but will use more memory.
        :param nonify: If all dataset should be nonized to the same dimension as the given value.

        :return: A Neighbours object
        """

        points = []

        if nonify:
            for dataset in self.datasets:
                dataset.nonify(value.dimension)

        values = [value for dataset in self.datasets for value in dataset.get_values()]

        for point in values:
            distance = self._distance(value, point, memoize)

            points.append(point.to_point(distance))

        return Neighbours(points)


