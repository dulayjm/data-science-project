"""
This file contains the random forest classifier.

The command line for this classifier takes one required argument and five optional arguments.

The required argument is:
    * split: a floating point number indicating the size of the split for data.

The optional arguments are:
    * criterion: the criterion (out of "gini" or "entropy") used to determine the splitting of nodes in decision trees.
    * max-depth: an integer indicating the max depth to which decision trees could continue to create nodes.
    * number-trees: an integer indicating the number of trees used in the random forest classifier.
    * seed: an integer representing the random seed which is used to indicate how the data will be shuffled.
    * shuffle: a boolean as to whether to shuffle the data after being loaded and before using it.
"""

from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms

from sklearn.ensemble import RandomForestClassifier
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.omniglot import Omniglot  # Currently not used below, but could be used.

from data.dataset import OmniglotReactionTimeDataset
from data.full_omniglot import FullOmniglot  # Currently not used below, but could be used.
from helpers.statistical_functions import *


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("split", type=float)
    parser.add_argument("--criterion", type=str, nargs="?", choices=["gini", "entropy"], default="gini")
    parser.add_argument("--max-depth", type=int, nargs="?", default=None)
    parser.add_argument("--number-trees", type=int, nargs="?", default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shuffle", type=bool, nargs="?", default=True)
    args = parser.parse_args()

    # You can add other transformations in the list below.
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = OmniglotReactionTimeDataset('sigma_dataset.csv', transforms=transform)

    if args.split <= 0 or args.split >= 1:
        raise ValueError("Invalid split value. Please provide a value between (but not equal to) 0 and 1.")

    dataset_size: int = len(dataset)
    indices: List[int] = list(range(dataset_size))
    split = int(np.floor(args.split * dataset_size))

    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=valid_sampler)

    dataiter = iter(train_loader)
    sample = dataiter.next()
    X = sample['image1']
    y = sample['label1']

    X = X.permute(1, 2, 3, 0)

    X = X.numpy()
    y = y.numpy()

    print('x shape', X.shape)
    print('y.shape', y.shape)

    dataiter = iter(validation_loader)
    sample_test = dataiter.next()
    X_test = sample_test['image1']
    y_test = sample_test['label1']

    X_test = X_test.permute(1, 2, 3, 0)

    X_test = X_test.numpy()
    y_test = y_test.numpy()

    X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3]).T
    y = y.reshape(y.shape[0],)

    X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1] * X_test.shape[2], X_test.shape[3]).T
    y_test = y_test.reshape(y_test.shape[0],)

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion=args.criterion,
                                 max_depth=args.max_depth, max_features='auto', max_leaf_nodes=None,
                                 min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                 n_estimators=args.number_trees, n_jobs=1, oob_score=False, random_state=None,
                                 verbose=0, warm_start=False)
    clf.fit(X, y)

    predictions = clf.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_base_statistics(predictions, y_test)
    display_base_statistics(args.seed, accuracy, precision, recall, f1_score)
