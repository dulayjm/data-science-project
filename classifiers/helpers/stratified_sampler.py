from random import sample
from typing import List

from torch.utils.data import Dataset, Sampler


# Trying to treat the below like BatchSampler.
# https://pytorch.org/docs/1.8.0/_modules/torch/utils/data/sampler.html#BatchSampler

class StratifiedKFoldSampler(Sampler[List[int]]):
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
        return sum([len(indices) for indices in self.folds])

    @staticmethod
    def _get_data_by_label(data_source) -> dict:
        data_by_label: dict = {}
        data_index: int = 0
        for item, label in data_source:
            if not data_by_label.get(label):
                data_by_label[label] = []
            data_by_label[label].append(data_index)
            data_index += 1
        return data_by_label

    def _compute_stratified_folds(self, data_source: Dataset) -> List[List[int]]:
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
        shifting_fold: List[int] = self.folds.pop(0)
        self.folds.append(shifting_fold)
        self.shifts += 1
