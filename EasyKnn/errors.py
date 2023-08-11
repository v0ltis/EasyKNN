class ValueAlreadyLinkedError(Exception):
    """Raised when trying to add a single value in two different datasets"""
    pass


class DatasetAlreadyLinkedError(Exception):
    """Raised when trying to add a dataset to two different plans"""
    pass


class NoDimensionError(Exception):
    """Raised when trying to add a value with no dimensions to a dataset"""
    pass


class ReadOnlyAttributeError(Exception):
    """Raised when trying to modify a read-only attribute"""
    pass


class CriticalDeletionError(Exception):
    """Raised when trying to delete a critical value"""
    pass
