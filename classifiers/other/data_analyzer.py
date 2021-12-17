import os

from argparse import ArgumentParser, FileType
from statistics import mean, stdev
from typing import List, Tuple

import torch

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Omniglot
from tqdm import tqdm

from data.dataset import OmniglotReactionTimeDataset
from data.full_omniglot import FullOmniglot
from data.data_preprocessors import preprocess_reaction_time_data


def get_data_by_class(classed_dataset: Dataset) -> dict:
    data_per_label: dict = {}
    for instance, instance_label in classed_dataset:
        if not data_per_label.get(instance_label):
            data_per_label[instance_label] = []
        data_per_label[instance_label].append(instance)
    return data_per_label


def calculate_average_representation(size: Tuple[int, int, int], tensors: List[Tensor]) -> Tensor:
    sum_tensor: Tensor = torch.tensor((), dtype=torch.float64)
    sum_tensor = Tensor.new_zeros(sum_tensor, size=size)
    for tensor in tensors:
        sum_tensor = sum_tensor.add(tensor)
    return sum_tensor.multiply(1 / len(tensors))


def calculate_overlap(fixed_tensor: Tensor, tensors: List[Tensor]) -> Tuple[float, float]:
    tensor_overlaps: List[float] = []
    for tensor in tensors:
        subtracted_tensor: Tensor = torch.abs(fixed_tensor.subtract(tensor))
        tensor_overlap: float = torch.sum(subtracted_tensor).item()
        tensor_overlaps.append(tensor_overlap)
    return mean(tensor_overlaps), stdev(tensor_overlaps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("analyses", type=str, nargs="+", choices=["overlap"])
    parser.add_argument("--data-source", type=str, choices=["full", "background", "evaluation", "reaction-time"],
                        default="reaction-time")
    parser.add_argument("--output-file", type=FileType(mode="w+", encoding="utf-8"), default="analyses_output.txt")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

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

    args.output_file.write("Data Analysis Results:\n")
    for arg in args.analyses:
        if arg == "overlap":
            args.output_file.write("Overlap Data:\n")
            data_by_label: dict = get_data_by_class(dataset)
            for label in tqdm(range(0, len(data_by_label.keys()))):
                args.output_file.write(f"\t* Class {label}:\n")
                data: list = data_by_label[label]
                class_average = calculate_average_representation(data[0].size(), data)
                average_overlap, overlap_stdev = calculate_overlap(class_average, data)
                args.output_file.write(f"\t\t> Average Overlap (with Standard Deviation): "
                                       f"{average_overlap} +/- {overlap_stdev}\n")
        else:
            raise ValueError(f"Value {arg} not recognized. Please try again.")
