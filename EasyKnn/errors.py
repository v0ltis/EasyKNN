class ValueAlreadyLinkedError(Exception):
    """Raised when trying to add a single :class:`Value<EasyKnn.value.Value>` in two different
    :class:`Datasets<EasyKnn.dataset.Dataset>`.

    >>> from EasyKnn import Value
    >>> from EasyKnn import Dataset
    >>>
    >>> dataset1 = Dataset()
    >>> dataset2 = Dataset()
    >>>
    >>> value = Value([1, 2, 3])
    >>>
    >>> dataset1.add_value(value)
    >>> dataset2.add_value(value)
    Traceback (most recent call last):
        ...
    EasyKnn.errors.ValueAlreadyLinkedError: A single value cannot be in two different datasets
    """
    pass


class DatasetAlreadyLinkedError(Exception):
    """Raised when trying to add a single :class:`Dataset<EasyKnn.dataset.Dataset>` to two different :class:`Plans<EasyKnn.plan.Plan>`.

    >>> from EasyKnn import Plan
    >>> from EasyKnn import Dataset
    >>>
    >>> plan1 = Plan()
    >>> plan2 = Plan()
    >>>
    >>> dataset = Dataset()
    >>>
    >>> plan1.add_dataset(dataset)
    >>> plan2.add_dataset(dataset)
    Traceback (most recent call last):
        ...
    EasyKnn.errors.DatasetAlreadyLinkedError: This dataset is already linked to a plan
    """
    pass


class NoDimensionError(Exception):
    """Raised when trying to create a :class:`Value<EasyKnn.value.Value> with no dimensions, or composed only of
    ``None``.

    >>> from EasyKnn import Value
    >>> value = Value([])
    Traceback (most recent call last):
        ...
    EasyKnn.errors.NoDimensionError: Coordinates cannot be empty or only None values

    >>> from EasyKnn import Value
    >>> value = Value([None, None, None])
    Traceback (most recent call last):
        ...
    EasyKnn.errors.NoDimensionError: Coordinates cannot be empty or only None values
    """
    pass


class ReadOnlyAttributeError(Exception):
    """Raised when trying to modify a read-only attribute

    >>> from EasyKnn import Value
    >>> value = Value([1, 2, 3])
    >>> print(value.dimension)
    3
    >>> value.dimension = 4
    Traceback (most recent call last):
        ...
    EasyKnn.errors.ReadOnlyAttributeError: The dimension attribute is read-only

    >>> from EasyKnn import Dataset, Plan
    >>> dataset = Dataset()
    >>> plan = Plan()
    >>>
    >>> plan.add_dataset(dataset)
    >>> plan.datasets = []
    Traceback (most recent call last):
        ...
    EasyKnn.errors.ReadOnlyAttributeError: The datasets attribute is read-only
    """
    pass


class CriticalDeletionError(Exception):
    """Raised when trying to delete a critical value

    >>> from EasyKnn import Value
    >>> value = Value([1, 2, 3])
    >>> del value.dimension
    Traceback (most recent call last):
        ...
    EasyKnn.errors.CriticalDeletionError: The dimension attribute cannot be deleted

    >>> from EasyKnn import Dataset
    >>> dataset = Dataset()
    >>> del dataset.linked_plan
    Traceback (most recent call last):
        ...
    EasyKnn.errors.CriticalDeletionError: The linked_plan attribute cannot be deleted
    """
    pass
