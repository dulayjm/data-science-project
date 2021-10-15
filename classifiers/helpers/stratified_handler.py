from random import sample

from torch.utils.data import Dataset


# Took a lot of inspiration from PyTorch's Sampler here;
# should this be treated like a BatchSampler?
# https://pytorch.org/docs/1.8.0/_modules/torch/utils/data/sampler.html#BatchSampler

class StratifiedKFoldHandler:
    def __init__(self, data_source: Dataset, num_folds: int = 10):
        # Ideally, I'd like to get rid of this value being stored here
        # unless it ends up being used in a more permanent fashion.
        self.data_source = data_source
        self.num_folds = num_folds
        self.folds: dict = self._compute_stratified_folds()

    # Since this is meant for k-fold cross-validation,
    # we don't want to mess with the order of sampling here.
    def __iter__(self):
        return iter(self.folds.items())

    def __len__(self):
        return sum([len(indices) for indices in self.folds.values()])

    def _get_data_by_label(self) -> dict:
        data_by_label: dict = {}
        data_index: int = 0
        for item, label in self.data_source:
            if not data_by_label.get(label):
                data_by_label[label] = []
            data_by_label[label].append(data_index)
            data_index += 1
        return data_by_label

    def _compute_stratified_folds(self) -> dict:
        folds: dict = {}
        label_frequencies: dict = self._get_data_by_label()
        for fold_number in range(0, self.num_folds):
            folds[fold_number] = []

        for label, indices in label_frequencies.items():
            # Here, we sample from the data in terms of its strata without replacement.
            # We do this via indices.
            fold_minimum: int = len(indices) // self.num_folds
            fold_sample_count: list = [fold_minimum for i in range(0, self.num_folds)]
            fold_remainder: int = len(indices) % self.num_folds
            if fold_remainder > 0:
                # We favor folds that have less items before randomly choosing.
                # However, if no folds have less items, then random selection is acceptable.
                # We run an algorithm at each step that determines which fold to add to.
                # It examines folds, generates a list of those with lower-than-maximum lengths,
                # and gives all those which are less than the maximum additional samples
                # (if such samples are available to be given).
                current_maximum_fold_size: int = max([len(indices) for indices in folds.values()])
                lesser_folds: list = [fold for fold, indices in folds.items()
                                      if len(indices) < current_maximum_fold_size]

                lesser_fold_sample_count: int = min([len(lesser_folds), fold_remainder])
                chosen_lesser_folds: list = sample(lesser_folds, lesser_fold_sample_count)
                for fold in chosen_lesser_folds:
                    fold_sample_count[fold] += 1
                    fold_remainder -= 1

                # If there are more folds and all fold counts are now equal,
                # then we sample randomly from all folds.
                if fold_remainder > 0:
                    winning_folds: list = sample(folds.keys(), fold_remainder)
                    for fold in winning_folds:
                        fold_sample_count[fold] += 1

            # Then, we perform actual random, stratified sampling without replacement.
            for fold, sample_count in enumerate(fold_sample_count):
                data_sample: list = sample(indices, sample_count)
                for item in data_sample:
                    indices.remove(item)
                folds[fold].extend(data_sample)

        return folds
