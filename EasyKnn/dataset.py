from typing import List

class Dataset:
    """
    Represent a set of data
    """

    def __init__(self):
        self._data = []

    def add_value(self, value: dict):
        self._data.append(value)

    def add_values(self, values: List[dict]):
        self._data.extend(values)

