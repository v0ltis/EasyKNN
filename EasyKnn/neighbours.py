from typing import List

from EasyKnn.errors import ReadOnlyAttributeError
from EasyKnn.dataset import Dataset
from EasyKnn.point import Point


class Neighbours:
    """
    Neighbours class represent the response of the Plan `neighbors` method.
    It is used to store the neighbours and distances of a given value.

    :param neighbours: A list of Point objects, representing the neighbours of the value.
    """
    def __init__(self, neighbours: List[Point]):

        self._neighbours = neighbours

        self._dataset_neighbours = []

        self._average_dist = sum([neighbour.distance for neighbour in self.neighbours]) / len(self.neighbours)

        self._process_data()

    @property
    def neighbours(self) -> List[Point]:
        """
        A list of all the neighbours of the value.

        :read-only: True
        """
        return self._neighbours

    @neighbours.setter
    @neighbours.deleter
    def neighbours(self, *args):
        raise ReadOnlyAttributeError("The neighbours attribute is read-only")

    @property
    def dataset_neighbours(self) -> List[Dataset]:
        """
        A list of all the datasets of the neighbours of the value.

        :read-only: True
        """
        return self._dataset_neighbours

    @dataset_neighbours.setter
    @dataset_neighbours.deleter
    def dataset_neighbours(self, *args):
        raise ReadOnlyAttributeError("The dataset_neighbours attribute is read-only")

    @property
    def average_dist(self) -> float:
        """
        The average distance of the neighbours to the value.

        :read-only: True
        """
        return self._average_dist

    @average_dist.setter
    @average_dist.deleter
    def average_dist(self, *args):
        raise ReadOnlyAttributeError("The average_dist attribute is read-only")

    def _process_data(self):
        """
        Process the data of the neighbours dataset, in order to obtain the nearest datasets.
        Also, sort the values, from the nearest to the farthest.

        Should only be called by the Plan class, and Once.

        :return:
        """

        # We start by ranking the datasets by the average distance of the neighbours
        count = {}

        for neighbour in self.neighbours:

            if neighbour.dataset in count:
                count[neighbour.dataset] += neighbour.distance

            else:
                count[neighbour.dataset] = neighbour.distance

        # We save each average distance in each dataset
        for dataset in count:
            dataset.average_dist = count[dataset] / len(dataset.data)

        #  We sort the datasets by the average distance

        self.dataset_neighbours = sorted(count.keys(), key=lambda x: x.average_dist)

        # We now sort the values by the distance

        self.neighbours = sorted(self.neighbours, key=lambda x: x.distance)

    def nearest_neighbour(self, k=1) -> List[Point]:
        """
        Get the k nearest neighbours of the neighbours. If k is negative, it will get the k farthest neighbours.
        If k > len(self.dataset_neighbours), it will return all the neighbours.

        :param k: The number of neighbours to get
        :return: The nearest neighbour of the neighbours
        """
        if k >= 0:

            return self.neighbours[:k]

        else:
            return self.neighbours[::-1][:-k]

    def nearest_dataset(self, k=1) -> List[Dataset]:
        """
        Get the k nearest datasets of the neighbours. If k is negative, it will get the k farthest datasets.
        If k > len(self.dataset_neighbours), it will return all the datasets.

        :param k: The number of neighbours to get
        :return: The nearest dataset of the neighbours
        """
        if k >= 0:
            return self.dataset_neighbours[:k]

        else:
            return self.dataset_neighbours[::-1][:-k]
