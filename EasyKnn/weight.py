from typing import List, Union

from EasyKnn.errors import ReadOnlyAttributeError, CriticalDeletionError


class Weight:
    """
    Weight represent the importance of a dimension in a value. It can be greater than 1 to make it more
    important, or less than 1 to make it less important. It can also be negative.
    Setting a weight of ``None`` will result as a weight of 1.

    A negative weight will subtract the value of the dimension to the distance, instead of adding it.
    If you set ``use_abs`` to ``False`` in :meth:`Plan.neighbors<EasyKnn.plan.Plan.neighbors>` method,
    negative values will be considered as nearest than a 0 distance to the searched value.

    :param weight: The weight of each dimension. Must be a ``list`` of ``None``, ``float`` or ``int``.
    """

    def __init__(self, weight: List[Union[float, int, None]] = None):

        if weight is None:
            weight = []

        self._weight = weight

        # We will check if the weight is only composed of Nones, floats and ints
        if not all([type(x) in [float, int, None] for x in weight]):
            raise TypeError("The weight must be composed of Nones, floats and ints")

    @property
    def weights(self) -> List[Union[float, int, None]]:
        """
        A ``list`` of weights, either ``None``, ``float`` or ``int``. Each weight
        represent the importance of a dimension.

        :read-only: True
        """

        return self._weight

    @weights.setter
    def weights(self, *args) -> None:
        raise ReadOnlyAttributeError("The weight attribute is read-only")

    @weights.deleter
    def weights(self, *args) -> None:
        raise CriticalDeletionError("The weight attribute cannot be deleted")

    def extend(self, data: Union[
        List[Union[float, int, None]],     # Either a list of weights
        Union[float, int, None]            # Or a single weight
    ]) -> None:
        """
        Extend the Weight for a new dimension. The new weight can be either a single weight or a list of weights.

        :param data: The new weight, either a single weight or a ``list`` of weights.
                     Weight(s) can be ``None``, ``float`` or ``int``.
        :return: ``None``
        """

        # We check if the given data is valid
        if type(data) == list:
            if not all([type(x) in [float, int, None] for x in data]):
                raise TypeError("The weight must be composed of Nones, floats and ints")
        elif type(data) not in [float, int, None]:
            raise TypeError("The weight must be either None, float and int")

        self._weight.extend(data)

    def __getitem__(self, item: int) -> Union[float, int, None]:
        if item >= len(self._weight):
            return 1

        value = self._weight[item]
        return value if value is not None else 1

    def __setitem__(self, key, value) -> None:

        # If the key is greater than the length of the weight, we raise an IndexError
        if key >= len(self._weight):
            raise IndexError("The index is out of range")

        # We check if the given data is valid
        elif type(value) not in [float, int, None]:
            raise TypeError("The weight must be either None, float and int")

        self._weight[key] = value

    def __delitem__(self, key) -> None:
        if key >= len(self._weight):
            raise IndexError("The index is out of range")

        self._weight[key] = None
