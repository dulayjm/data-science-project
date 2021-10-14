import os

from argparse import ArgumentParser
from random import choice
from typing import List

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets
from torchvision import transforms


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_location", type=str)
    args = parser.parse_args()

    if not (args.data_location and os.path.exists(args.data_location) and os.path.isdir(args.data_location)):
        data_location: str = os.getcwd()
    else:
        data_location: str = args.data_location

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Retrieve data from dataset:
    background_dataset = datasets.Omniglot(data_location, background=True, download=True, transform=transform)
    evaluation_dataset = datasets.Omniglot(data_location, background=False, download=True, transform=transform)

    # Accumulate the labels:
    labels: List[int] = []
    for image, label in background_dataset:
        if label not in labels:
            labels.append(label)

    label_offset: int = labels[-1] + 1
    for image, label in evaluation_dataset:
        if (label + label_offset) not in labels:
            labels.append(label + label_offset)

    # Perform random predictions:
    ground_truth: list = []
    predictions: list = []
    for image, label in background_dataset:
        predicted_label: int = choice(labels)
        predictions.append(predicted_label)
        ground_truth.append(label)

    for image, label in evaluation_dataset:
        predicted_label: int = choice(labels)
        predictions.append(predicted_label)
        ground_truth.append(label + label_offset)

    accuracy: float = accuracy_score(ground_truth, predictions)
    precision: float = precision_score(ground_truth, predictions, average='macro')
    recall: float = recall_score(ground_truth, predictions, average='macro')
    f1_score: float = f1_score(ground_truth, predictions, average='macro')

    # TODO: add random seed to result output?
    print(f"Results:\n"
          f"\t* Accuracy: {accuracy}\n"
          f"\t* Precision: {precision}\n"
          f"\t* Recall: {recall}\n"
          f"\t* F1 Score: {f1_score}\n")
