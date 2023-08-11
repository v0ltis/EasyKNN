from typing import List

from EasyKnn.errors import ReadOnlyAttributeError, CriticalDeletionError
from EasyKnn.dataset import Dataset
from EasyKnn.point import Point


class Neighbors:
    """
    Neighbors class represent the response of the :meth:`Plan<EasyKnn.plan.Plan.neighbors>` method.
    It is used to store the neighbors and distances of a given :class:`Value<EasyKnn.value.Value>`.

    :param neighbors: A list of :class:`Points<EasyKnn.point.Point>`, representing the neighbors of the value.
    """
    def __init__(self, neighbors: List[Point]):

        self._neighbors = neighbors

        self._dataset_neighbors = []

        self._average_dist = sum([neighbor.distance for neighbor in self.neighbors]) / len(self.neighbors)

        self._process_data()

    @property
    def neighbors(self) -> List[Point]:
        """
        A ``list`` of all the :class:`Point<EasyKnn.point.Point>` in all the
        :class:`Datasets<EasyKnn.dataset.Dataset>` from the :class:`Plan<EasyKnn.plan.Plan>`.

        :read-only: True
        """
        return self._neighbors

    @neighbors.setter
    def neighbors(self, *args):
        raise ReadOnlyAttributeError("The neighbors attribute is read-only")

    @neighbors.deleter
    def neighbors(self, *args):
        raise CriticalDeletionError("The neighbors attribute cannot be deleted")

    @property
    def dataset_neighbors(self) -> List[Dataset]:
        """
        A ``list`` of all the :class:`Datasets<EasyKnn.dataset.Dataset>` used in the :class:`Plan<EasyKnn.plan.Plan>`.

        :read-only: True
        """
        return self._dataset_neighbors

    @dataset_neighbors.setter
    def dataset_neighbors(self, *args):
        raise ReadOnlyAttributeError("The dataset_neighbors attribute is read-only")

    @dataset_neighbors.deleter
    def dataset_neighbors(self, *args):
        raise CriticalDeletionError("The dataset_neighbors attribute cannot be deleted")

    @property
    def average_dist(self) -> float:
        """
        The average distance between each :class:`Points<EasyKnn.point.Point>` and the
        :class:`Value<EasyKnn.value.Value>`.

        :read-only: True
        """
        return self._average_dist

    @average_dist.setter
    def average_dist(self, *args):
        raise ReadOnlyAttributeError("The average_dist attribute is read-only")

    @average_dist.deleter
    def average_dist(self, *args):
        raise CriticalDeletionError("The average_dist attribute cannot be deleted")

    def _process_data(self) -> None:
        """
        Process the data of all the :attr:`Plan.datasets<EasyKnn.plan.Plan.Datasets>`, in order to obtain the nearest datasets.
        Also, sort the values, from the nearest to the farthest.

        :return: ``None``
        """

        # We start by ranking the datasets by the average distance of the neighbors
        count = {}

        for neighbor in self.neighbors:

            if neighbor.dataset in count:
                count[neighbor.dataset] += neighbor.distance

            else:
                count[neighbor.dataset] = neighbor.distance

        # We save each average distance in each dataset
        for dataset in count:
            dataset._average_dist = count[dataset] / len(dataset.data)

        #  We sort the datasets by the average distance

        self._dataset_neighbors = sorted(count.keys(), key=lambda x: x.average_dist)

        # We now sort the values by the distance

        self._neighbors = sorted(self.neighbors, key=lambda x: x.distance)

    def nearest_neighbor(self, k: int = 1) -> List[Point]:
        """
        Get the ``k`` nearest :class:`Point<EasyKnn.point.Point>` of the value. If ``k`` is negative, it will get the
        ``k`` farthest neighbors. If ``k`` is greater than the number of neighbors, it will return all the neighbors.

        :param k: The number of neighbors to get. Default is 1
        :return: The ``k`` nearest :class:`Points<EasyKnn.point.Point>` of the :class:`Value<EasyKnn.value.Value>`.
        """
        if k >= 0:

            return self.neighbors[:k]

        else:
            return self.neighbors[::-1][:-k]

    def nearest_dataset(self, k: int = 1) -> List[Dataset]:
        """
        Get the ``k`` nearest :class:`Datasets<EasyKnn.dataset.Dataset>` of the value. If ``k`` is negative, it will get
        the ``k`` farthest datasets. If ``k`` is greater than the number of datasets, it will return all the datasets.

        :param k: The number of neighbors to get. Default is 1
        :return: The ``k`` nearest :class:`Datasets<EasyKnn.dataset.Dataset>` of the :class:`Value<EasyKnn.value.Value>`.
        """
        if k >= 0:
            return self.dataset_neighbors[:k]

        else:
            return self.dataset_neighbors[::-1][:-k]
