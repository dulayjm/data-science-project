"""
This file contains the KNN classifier.

It requires arguments of the number of neighbors used in classification (k)
and the split value for the type of split employed (split).

It takes optional arguments of the data source, distance type, random seed, shuffle boolean, split type,
and tensor transforms given below.
    * data-source: the source from which the data is taken. There are four sources:
        (1) full: the whole Omniglot dataset, consisting of both the background and evaluation sets.
        (2) background: the background Omniglot dataset, taken from the Omniglot class in PyTorch.
        (3) evaluation: the evaluation Omniglot dataset, taken from the Omniglot class in PyTorch.
        (4) reaction-time: the reaction time dataset, which was made by previous work
        and consists of less classes and examples.
    * distance-type: the kind of distance used to determine how close data objects are to one another.
    Currently, the only option is CosineSimilarity, although other functions could be added.
    * seed: the value which is used to seed the sampling process.
    * shuffle: a boolean as to whether to shuffle the data after being loaded and before using it.
    Here, this applies in the case of random splits for the data.
    * split-type: the type of fold-splitting employed, having options of "none", "random", or "stratified".
    * transforms: functions to apply to the input tensors. "raw" refers to having no operations performed.
    "resized" shrinks the input tensors to 28-by-28 tensors if they are larger than this size.
    Finally, "flattened" turns a tensor into a vector. Both 'resized" and "flattened" can be applied at the same time.
"""

import os
import random

from argparse import ArgumentParser
from random import choice, seed
from heapq import nlargest
from typing import Callable, List, Tuple

from aenum import NamedTuple
from torch.nn import CosineSimilarity
from torchvision.datasets import Omniglot
from torchvision import transforms
from tqdm import tqdm

from data.full_omniglot import FullOmniglot
from data.dataset import OmniglotReactionTimeDataset
from data.data_preprocessors import preprocess_reaction_time_data
from helpers.fold_splitter import split_folds
from helpers.statistical_functions import *
from helpers.stratified_sampler import StratifiedKFoldSampler

# Constants:
MIN_NEIGHBORS: int = 1
MAX_NEIGHBORS: int = 32639
RESIZED_SIZE: Tuple[int, int] = (28, 28)
RANDOM_MINIMUM: int = 0
RANDOM_MAXIMUM: int = 2**31 - 1

# NamedTuples:
LabeledScore: NamedTuple = NamedTuple("LabeledScore", "score label")


def get_nearest_neighbor(nearest_labels: list) -> int:
    label_frequencies: dict = {}
    for (score, label) in nearest_labels:
        if not label_frequencies.get(label):
            label_frequencies[label] = 0
        label_frequencies[label] += 1

    frequency_labels: dict = {frequency: [] for frequency in label_frequencies.values()}
    for label, frequency in label_frequencies.items():
        frequency_labels[frequency].append(label)

    maximum_frequency: int = max([frequency for frequency in frequency_labels.keys()])
    # If we have a tie, we choose a label randomly. Otherwise, we have the winning label.
    if len(frequency_labels[maximum_frequency]) > 1:
        winning_label: int = choice(frequency_labels[maximum_frequency])
    else:
        winning_label: int = frequency_labels[maximum_frequency][0]

    return winning_label


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("k", type=int, choices=range(MIN_NEIGHBORS, MAX_NEIGHBORS))
    parser.add_argument("split", type=int)
    parser.add_argument("--data-source", type=str, choices=["full", "background", "evaluation", "reaction-time"],
                        default="reaction-time")
    parser.add_argument("--distance-type", type=str, choices=["cosine"], default="cosine")
    parser.add_argument("--seed", type=int, default=random.randint(RANDOM_MINIMUM, RANDOM_MAXIMUM))
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--split-type", type=str, choices=["random", "stratified"], default="random")
    parser.add_argument("--transforms", type=str, nargs="*",
                        choices=["raw", "resized", "flattened"], default="raw")  # TODO: make multi-arg?
    args = parser.parse_args()  # TODO: investigate subparsers?

    if args.seed:
        if args.seed >= 0:
            seed(args.seed)
        else:
            raise ValueError("Invalid seed. Please try again.")

    # We perform data validation for various options here:
    if args.distance_type == "cosine":
        distance_function: Callable = CosineSimilarity(dim=0)
    else:
        raise ValueError("Invalid distance type given. Please try again.")

    if args.split_type == "stratified" or args.split_type == "random":
        if args.split <= 1:
            raise ValueError("Invalid number of folds. Please provide a value greater than one "
                             "and smaller than the data size.")

    # Retrieve data from dataset:
    transforms_list: list = [transforms.ToTensor()]
    for transform_name in args.transforms:
        if transform_name == "resized":
            transforms_list.append(transforms.Resize(RESIZED_SIZE))
        if transform_name == "flattened":
            transforms_list.append(transforms.Lambda(lambda t: t.flatten()))
    transform = transforms.Compose(transforms_list)

    if args.data_source == "full":
        dataset = FullOmniglot(os.getcwd(), transform=transform)
    elif args.data_source == "background":
        dataset = Omniglot(os.getcwd(), transform=transform)
    elif args.data_source == "evaluation":
        dataset = Omniglot(os.getcwd(), background=False, transform=transform)
    elif args.data_source == "reaction-time":
        dataset = preprocess_reaction_time_data(OmniglotReactionTimeDataset('sigma_dataset.csv', transforms=transform))
    else:
        raise ValueError("Appropriate dataset not specified. Please try again with one of the possible options.")

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    if args.split_type == "stratified" or args.split_type == "random":
        if args.split_type == "stratified":
            folds: List[List[int]] = [fold for fold in StratifiedKFoldSampler(dataset, args.split)]
        else:  # that is, args.split_type == "random"
            indices: List[int] = list(range(len(dataset)))
            folds: List[List[int]] = split_folds(indices, args.split, should_shuffle=args.shuffle)

        for fold_number in range(0, args.split):
            training_set: list = [index for fold in folds[1:] for index in fold]
            test_set: list = folds.pop(0)

            # Once we have the training and the test data, we can begin the algorithm.
            # In particular, we use the distance_function to compute distance between each item.
            # Then, we find the nlargest items in the array. We use the majority of their classes to classify the item.
            # If there is no majority, we randomize between the most frequent classes.
            # TODO: this is a little simplistic. We could do multiple things here:
            # we could take the label with the best score,
            # or we could take the label with the best average score from those items with the highest frequency...

            ground_truth: list = []
            predictions: list = []
            for test_index in tqdm(test_set):
                test_tensor, test_label = dataset[test_index]
                ground_truth.append(test_label)
                labeled_scores: list = []
                for training_index in training_set:
                    training_tensor, training_label = dataset[training_index]

                    # If, for some reason, there is an issue with the tensor size,
                    # this avoids error. A couple images were,
                    # for some reason, not having "3" as their third dimension.
                    # Instead, such tensors had "1"; these were few and far between, however.
                    if test_tensor.size() != training_tensor.size():
                        continue
                    labeled_scores.append(LabeledScore(distance_function(test_tensor, training_tensor), training_label))

                maximum_labeled_scores = nlargest(n=args.k, iterable=labeled_scores, key=lambda x: x.score)
                predicted_label = get_nearest_neighbor(maximum_labeled_scores)
                predictions.append(predicted_label)

            # We collect and store summative results...
            accuracy, precision, recall, f1_score = calculate_base_statistics(predictions, ground_truth)
            display_base_statistics(args.seed, accuracy, precision, recall, f1_score, fold_number + 1)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

            # We clear the predictions and ground truth...
            predictions.clear()
            ground_truth.clear()

            # We iterate by putting the test set back at the end of the folds.
            folds.append(test_set)
        else:
            # Post-looping for the folds, we calculate and display overall statistics.
            distributions: list = calculate_fold_statistics(accuracies, precisions, recalls, f1_scores)
            display_fold_statistics(args.seed, args.split_value, *distributions)
    else:
        raise NotImplementedError("Random sampling has not yet been implemented for this task.")
