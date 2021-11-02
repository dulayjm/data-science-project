import os

from argparse import ArgumentParser
from random import choice, seed
from typing import List

from torchvision.datasets import Omniglot
from torchvision import transforms
from tqdm import tqdm

from data.dataset import OmniglotReactionTimeDataset
from data.full_omniglot import FullOmniglot
from helpers.stratified_sampler import StratifiedKFoldSampler
from helpers.statistical_functions import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_source", type=str, choices=["full", "background", "evaluation", "reaction-time"],
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
    if args.data_source == "full":
        dataset = FullOmniglot(os.getcwd(), transform=transform)
    elif args.data_source == "background":
        dataset = Omniglot(os.getcwd(), transform=transform)
    elif args.data_source == "evaluation":
        dataset = Omniglot(os.getcwd(), background=False, transform=transform)
    elif args.data_source == "reaction-time":
        dataset = OmniglotReactionTimeDataset('../sigma_dataset.csv', transforms=transform)
    else:
        raise ValueError("Appropriate dataset not specified. Please try again with one of the possible options.")

    # TODO: add this to perform random trials under stratified condition
    if args.split_type == "stratified":
        folds = [fold for fold in StratifiedKFoldSampler(dataset, int(args.split_value))]

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
            print(fold)
            for index in fold:
                predicted_label: int = choice(labels)
                predictions.append(predicted_label)
                image, label = dataset[index]
                ground_truth.append(label)
            accuracy, precision, recall, f_score = calculate_base_statistics(predictions, ground_truth)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f_score)
            display_base_statistics(args.seed, accuracy, precision, recall, f_score, fold_number)
            predictions.clear()
            ground_truth.clear()
        else:
            distributions: list = calculate_fold_statistics(accuracies, precisions, recalls, f1_scores)
            display_fold_statistics(args.seed, args.split_value, *distributions)
    elif args.split_type == "none":
        for image, label in tqdm(dataset):
            predicted_label: int = choice(labels)
            predictions.append(predicted_label)
            ground_truth.append(label)
        accuracy, precision, recall, f_score = calculate_base_statistics(predictions, ground_truth)
        display_base_statistics(args.seed, accuracy, precision, recall, f_score)
