"""
This file contains a function that is useful for creating random folds from data.
See the documentation on the function below.
"""

from random import shuffle
from typing import List


def split_folds(indices: List[int], num_folds: int, should_shuffle: bool = True) -> List[List[int]]:
    """
    This function takes a set of indices and divides them into folds.
    It does so in a simple manner, doling them out to each fold incrementally.
    It adds one item to each fold in sequential order so long as there are indices left which have not been added.
    Args:
        - indices: the list of indices (List[int]) which can be used to refer to all data objects in the given data.
        - num_folds: the number of folds (int) into which the indices should be divided.
        Whether or not the data is divided evenly between folds has no bearing on the matter.
        - should_shuffle: a boolean (bool) determining whether or not the indices
        should be shuffled via random.shuffle() before use.
    Returns: a list of lists of integer indices (List[List[int]]) representing individual folds.
    All indices will only be a part of a single list.
    """
    folds: List[List[int]] = []
    for fold in range(0, num_folds):
        folds.append([])

    if should_shuffle:
        shuffle(indices)

    fold_index: int = 0
    for index in indices:
        folds[fold_index].append(index)
        fold_index += 1
        if fold_index >= num_folds:
            fold_index = 0

    return folds
