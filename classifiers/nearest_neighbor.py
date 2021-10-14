import os

from argparse import ArgumentParser, BooleanOptionalAction

from torchvision import datasets
from torchvision import transforms

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("k", type=int)
    parser.add_argument("--split", type=float)
    parser.add_argument("--split-type", type=str, choices=["stratified"])
    parser.add_argument("--shuffle", type=bool, )
    parser.add_argument("--data_location", type=str, action=BooleanOptionalAction)
    args = parser.parse_args()

    if not (os.path.exists(args.data_location) or os.path.isdir(args.data_location)):
        data_location: str = os.getcwd()
    else:
        data_location: str = args.data_location

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Retrieve data from dataset:
    dataset = datasets.Omniglot(data_location, download=True, transform=transform)
