"""
This file contains the random classifier.

This classifier takes four optional arguments:
    * data-source: the source from which the data is taken. There are four sources:
        (1) full: the whole Omniglot dataset, consisting of both the background and evaluation sets.
        (2) background: the background Omniglot dataset, taken from the Omniglot class in PyTorch.
        (3) evaluation: the evaluation Omniglot dataset, taken from the Omniglot class in PyTorch.
        (4) reaction-time: the reaction time dataset, which was made by previous work
        and consists of less classes and examples.
    * seed: the value which is used to seed the sampling process.
    * split-type: the type of fold-splitting employed, having options of "none", "random", or "stratified".
    * split-value: a floating point number indicating how the split will be performed,
    either subdividing the dataset in some way (0 < x < 1) or indicating the number of folds (x > 1).
"""

import os

from argparse import ArgumentParser
from random import choice, seed
from typing import Callable, List

from torchvision.datasets import Omniglot
from torchvision import transforms
from tqdm import tqdm

from data.dataset import OmniglotReactionTimeDataset
from data.full_omniglot import FullOmniglot
from data.data_preprocessors import preprocess_reaction_time_data
from helpers.stratified_sampler import StratifiedKFoldSampler
from helpers.statistical_functions import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-source", type=str, choices=["full", "background", "evaluation", "reaction-time"],
                        default="full")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-type", type=str, choices=["none", "random", "stratified"], default="none")
    parser.add_argument("--split-value", type=float)
    args = parser.parse_args()

    print(f"Settings: Data Source - {args.data_source}; "
          f"Seed: {args.seed if args.seed else 'None'}; "
          f"Split Type: {args.split_type}; "
          f"Split Value: {args.split_value if args.split_value else 'None'}")

    if args.seed:
        if args.seed >= 0:
            seed(args.seed)
        else:
            raise ValueError("Invalid seed. Please try again.")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Retrieve data from dataset:
    data_preprocessor: Union[Callable, None] = None
    if args.data_source == "full":
        dataset = FullOmniglot(os.getcwd(), transform=transform)
    elif args.data_source == "background":
        dataset = Omniglot(os.getcwd(), transform=transform)
    elif args.data_source == "evaluation":
        dataset = Omniglot(os.getcwd(), background=False, transform=transform)
    elif args.data_source == "reaction-time":
        dataset = preprocess_reaction_time_data(
            OmniglotReactionTimeDataset('sigma_dataset.csv', transforms=transform)
        )
    else:
        raise ValueError("Appropriate dataset not specified. Please try again with one of the possible options.")

    if args.split_type == "stratified":
        folds = [fold for fold in
                 StratifiedKFoldSampler(dataset, int(args.split_value))]

    # Accumulate the labels:
    # TODO: there is an assumption here that "stratified" contains evenly-split data throughout all folds.
    # However, it is possible that, in some circumstances, "stratified" won't get an instance of a class.
    # For our problem, it does not yet seem relevant, but this is a limitation of the code.
    if args.split_type in ["none", "stratified"]:
        labels: List[int] = []
        for image, label in tqdm(dataset):
            if label not in labels:
                labels.append(label)
    else:
        raise NotImplementedError("Random splits not yet implemented.")

    # Perform random predictions, reseeding seed() now that the static part of the seed is used up.
    seed()
    ground_truth: list = []
    predictions: list = []

    if args.split_type == "stratified" and folds is not None:
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        for fold_number, fold in tqdm(enumerate(folds, start=1)):
            for index in fold:
                predicted_label: int = choice(labels)
                predictions.append(predicted_label)
                image, label = dataset[index]
                ground_truth.append(label)

            # The below tabulates results and displays them:
            accuracy, precision, recall, f_score = calculate_base_statistics(predictions, ground_truth)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f_score)
            display_base_statistics(args.seed, accuracy, precision, recall, f_score, fold_number)

            # Finally, we clear and reuse the lists.
            predictions.clear()
            ground_truth.clear()
        else:
            # Post-looping for the folds, we calculate and display overall statistics.
            distributions: list = calculate_fold_statistics(accuracies, precisions, recalls, f1_scores)
            display_fold_statistics(args.seed, args.split_value, *distributions)
    elif args.split_type == "none":
        for image, label in tqdm(dataset):
            predicted_label: int = choice(labels)
            predictions.append(predicted_label)
            ground_truth.append(label)
        accuracy, precision, recall, f_score = calculate_base_statistics(predictions, ground_truth)
        display_base_statistics(args.seed, accuracy, precision, recall, f_score)
