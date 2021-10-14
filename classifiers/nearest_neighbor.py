import os

from argparse import ArgumentParser
from heapq import nlargest
from typing import Callable

from torch.nn import CosineSimilarity
from torchvision.datasets import Omniglot
from torchvision import transforms

from helpers.full_omniglot import FullOmniglot

# Constants:
MIN_NEIGHBORS: int = 1
MAX_NEIGHBORS: int = 32639


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("k", type=int, choices=range(MIN_NEIGHBORS, MAX_NEIGHBORS))
    parser.add_argument("split", type=float)
    parser.add_argument("--distance-type", type=str, choices=["cosine"], default="cosine")
    parser.add_argument("--split-type", type=str, choices=["random", "stratified"], default="random")
    parser.add_argument("--shuffle", type=bool)
    parser.add_argument("--data_location", type=str)
    args = parser.parse_args()

    # We perform data validation for various options here:

    if args.distance_type == "cosine":
        distance_function: Callable = CosineSimilarity
    else:
        raise ValueError("Invalid distance type given. Please try again.")

    if args.split <= 0.0 or args.split >= 1.0:
        raise ValueError("Invalid split percentage. Please provide a value within the range of (0, 1).")

    if not (args.data_location and os.path.exists(args.data_location) and os.path.isdir(args.data_location)):
        data_location: str = os.getcwd()
    else:
        data_location: str = args.data_location

    # Retrieve data from dataset:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # background_dataset = Omniglot(data_location, background=True, download=True, transform=transform)
    #
    # unique_labels: int = len([label for image, label in background_dataset])
    #
    # target_transform = transforms.Compose([
    #     lambda value: value + len(unique([label for image, label in background_dataset])) + 1
    # ])
    # evaluation_dataset = Omniglot(data_location, background=False, download=True, transform=transform,
    #                               target_transform=target_transform)

    full_dataset = FullOmniglot(data_location, transform=transform)

    # First, we divide up the data into its component parts.
    ...  # TODO: implement stratified .

    # Once we have the training and the test data, we can begin the algorithm.
    # In particular, we use the distance_function to compute distance between each item.
    # Then, we find the nlargest items in the array. We use the majority of their classes to classify the item.
    # If there is no majority, we elect to use the nearest item. If there is a tie for the nearest, we randomize.
    ...  # TODO: implement nearest-neighbor using the distance function and heapq's nlargest.

    # Finally, we score the results.
    ...  # TODO: implement scoring.