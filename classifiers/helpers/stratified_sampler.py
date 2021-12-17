"""
This file contains the StratifiedKFoldSampler that was built for this project.
It had been used to an extent, but it was not ready enough to feature more prominently in the project.
Still, it seems to have reached functionality by this point, and it can thus be employed in future work.
"""

from random import sample
from typing import List

from torch.utils.data import Dataset, Sampler


# In building this class, we attempted to treat the below like PyTorch's BatchSampler:
# https://pytorch.org/docs/1.8.0/_modules/torch/utils/data/sampler.html#BatchSampler

class StratifiedKFoldSampler(Sampler[List[int]]):
    """
    This class inherits from PyTorch's Sampler. It attempts to provide stratified, k-fold sampling.
    It has a few functions to aid in processing. See them below for more detail.
    The one function able to be used in aiding with k-fold cross-validation is shift(),
    which is meant to move the fold at the front of the list to its back.
    This would mean that, on the next use of __iter__, the order of folds would be different.
    It is useful particularly when moving between stages of cross-validation.

    Args:
        - data_source: the dataset from which data is drawn. In particular,
        this dataset needs to be able to unload the data object and its label via __iter__.
        - num_founds: the number of folds into which the data will be split.
        Note that the code below has no restrictions on perfectly-even splits.
        In other words, if there are 100 objects and 9 folds are desired,
        the code below will generate folds as evenly as possible.
    """

    def __init__(self, data_source: Dataset, num_folds: int = 10):
        # The typing complaint here is of no concern; Dataset is Sized for all of our uses.
        super().__init__(data_source)  # type: ignore
        self.num_folds = num_folds
        self.shifts = 0
        self.folds: List[List[int]] = self._compute_stratified_folds(data_source)

    # Since this is meant for k-fold cross-validation,
    # we don't want to mess with the order of sampling here.
    def __iter__(self):
        for fold in self.folds:
            yield fold

    def __len__(self):
        """
        In this case, we have set __len__ to return the total number of objects in the data.
        If this code were expanded upon, it might be beneficial to consider adding accessors for,
        say, the number of folds, the number of shifts, and so on.

        Returns: total number of items in all folds (int).
        """
        return sum([len(indices) for indices in self.folds])

    @staticmethod
    def _get_data_by_label(data_source: Dataset) -> dict:
        """
        This function sorts the dataset by its labels. It returns a dictionary,
        where the labels are keys and a list of the pertinent data points are values.
        Currently, this only handles data which contains a single label;
        it cannot handle multi-label classification.

        Args:
            - data_source: the dataset from which data is drawn. In particular,
        this dataset needs to be able to unload the data object and its label via __iter__.
        """
        data_by_label: dict = {}
        data_index: int = 0
        for item, label in data_source:
            if not data_by_label.get(label):
                data_by_label[label] = []
            data_by_label[label].append(data_index)
            data_index += 1
        return data_by_label

    def _compute_stratified_folds(self, data_source: Dataset) -> List[List[int]]:
        """
        This function forms the stratified folds. It utilizes the _get_data_by_label function
        to sort the data by its labels, and it then goes through all labels and divides up the pertinent data
        among the folds. It does its best to divide the data up evenly, and it keeps track of
        both the sizes of the folds and the number of data items allocated to each fold per class.
        In other words, the system can handle uneven class sizes and does its best to accommodate for this.
        In particular, the result will have folds as evenly distributed in terms of size as possible.
        Args:
            - data_source: the dataset from which data is drawn. In particular,
            this dataset needs to be able to unload the data object and its label via __iter__.
        Returns: the folds into which the items in data_source have been provided. In particular,
        all classes are stratified as is described above.
        """
        folds: List[List[int]] = []
        label_frequencies: dict = self._get_data_by_label(data_source)  # type: ignore
        # We pre-populate the data structure with empty lists.
        for fold_number in range(0, self.num_folds):
            folds.append([])

        for label, indices in label_frequencies.items():
            # Here, we sample from the data in terms of its strata without replacement.
            # We do this via indices.
            fold_minimum: int = len(indices) // self.num_folds
            fold_sample_count: list = [fold_minimum for _ in range(0, self.num_folds)]
            fold_remainder: int = len(indices) % self.num_folds
            if fold_remainder > 0:
                # We favor folds that have less items before randomly choosing.
                # However, if no folds have less items, then random selection is acceptable.
                # We run an algorithm at each step that determines which fold to add to.
                # It examines folds, generates a list of those with lower-than-maximum lengths,
                # and gives all those which are less than the maximum additional samples
                # (if such samples are available to be given).
                current_maximum_fold_size: int = max([len(indices) for indices in folds])
                lesser_folds: list = [fold for fold, indices in enumerate(folds)
                                      if len(indices) < current_maximum_fold_size]

                lesser_fold_sample_count: int = min([len(lesser_folds), fold_remainder])
                chosen_lesser_folds: list = sample(lesser_folds, lesser_fold_sample_count)
                for fold in chosen_lesser_folds:
                    fold_sample_count[fold] += 1
                    fold_remainder -= 1

                # If there are more folds and all fold counts are now equal,
                # then we sample randomly from all folds.
                if fold_remainder > 0:
                    winning_folds: list = sample([fold for fold in range(0, self.num_folds)], fold_remainder)
                    for fold in winning_folds:
                        fold_sample_count[fold] += 1

            # Then, we perform actual random, stratified sampling without replacement.
            for fold, sample_count in enumerate(fold_sample_count):
                data_sample: list = sample(indices, sample_count)
                for item in data_sample:
                    indices.remove(item)
                folds[fold].extend(data_sample)

        return folds

    def shift(self):
        """
        This function shifts the first fold to the end of the list.
        This is mainly for use in performing cross-validation,
        as it can be used to produce the next sequence of folds for a new iteration once one is done.
        It also keeps track of the number of shifts by incrementing the self.shifts field.
        """
        shifting_fold: List[int] = self.folds.pop(0)
        self.folds.append(shifting_fold)
        self.shifts += 1
