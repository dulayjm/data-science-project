"""
This file contains the Siamese network.

Currently, it takes one argument and six optional arguments.

The non-optional argument is:
    * split: a floating point number between 0 and 1 which indicates the way in which the data
    will be split for training and testing.

The optional arguments are:
    * batch-size: the size of a given batch--in other words, the number of data objects that are part of each batch.
    * epochs: the number of epochs for which the neural network will be trained.
    * learning-rate: a floating point number indicating the learning rate for the neural network.
    * momentum: a floating point number indicating the momentum factor for stochastic gradient descent.
    * seed: an integer representing the random seed which is used to indicate how the data will be shuffled.
    * shuffle: a boolean as to whether to shuffle the data after being loaded and before using it.
"""

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.omniglot import Omniglot  # Currently not used below, but could be used.

from data.dataset import OmniglotReactionTimeDataset
from data.full_omniglot import FullOmniglot  # Currently not used below, but could be used.


class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),       # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),                  # 128@42*42
            nn.MaxPool2d(2),            # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),                  # 128@18*18
            nn.MaxPool2d(2),            # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),                  # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("split", type=float)
    parser.add_argument("--batch-size", type=int, nargs="?", default=16)
    parser.add_argument("--epochs", type=int, nargs="?", default=2)
    parser.add_argument("--learning-rate", type=float, nargs="?", default=.001)
    parser.add_argument("--momentum", type=float, nargs="?", default=0.9)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shuffle", type=bool, nargs="?", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # You can add other transformations to this list.
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # dataset = torchvision.datasets.Omniglot(os.getcwd(), download=True, transform=transform)

    dataset = OmniglotReactionTimeDataset('sigma_dataset.csv', transforms=transform)

    if args.split <= 0 or args.split >= 1:
        raise ValueError("Invalid split value. Please provide a value between (but not equal to) 0 and 1.")

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.split * dataset_size))

    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)

    # Model Work:
    model = SiameseNetwork()

    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        for idx, sample in enumerate(train_loader):
            image1 = sample['image1']
            image2 = sample['image2']

            label1 = sample['label1']
            label2 = sample['label2']

            labels = label1.to(device)

            # concatenate the batched images for now
            image1 = image1.to(device)
            image2 = image2.to(device)

            # may not need to concat, and just run them separately ...
            outputs = model(image1, image2).to(device)
            loss = criterion(outputs, label1)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()

            # this seemed to fix the accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Finished Training')
