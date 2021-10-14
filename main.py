import argparse
import os
import time

from torch.utils.data import dataset

import neptune
import numpy as np
from numpy.random.mtrand import seed
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet50

from dataset import OmniglotReactionTimeDataset
from classifiers.helpers.psychloss import AccPsychCrossEntropyLoss, PsychCrossEntropyLoss

# critical args
parser = argparse.ArgumentParser(description='Training Psych Loss.')

parser.add_argument('--dataset_type', type=str,
                    help='type of dataset to use. should be torch or custom')
parser.add_argument('--task', type=str,
                    help='type of task to perform. WIP. should be svm, randomforest, dl, unsupervised')

# dataset modification args
parser.add_argument('--transform', type=str, default=None,
                    help='torchvision transforms on the dataset')
parser.add_argument('--to_numpy', type=bool, default=False,
                    help='convert dataset to numpy. needed for classical ML classifiers')

# deep learnings args
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs to use')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--num_classes', type=int, default=100,
                    help='number of classes')
parser.add_argument('--learning_rate', type=float, default=0.01, 
                    help='learning rate')
parser.add_argument('--loss_fn', type=str, default='psych-rt',
                    help='loss function to use. select: cross-entropy, psych-rt, psych-acc')                 
parser.add_argument('--dataset_file', type=str, default='small_sigma.csv',
                    help='dataset file to use. out.csv is the full set')
parser.add_argument('--use_neptune', type=bool, default=False,
                    help='log metrics via neptune')

args = parser.parse_args()

assert args.dataset_type is not None, 'please specify dataset type'
assert args.task is not None, 'please specify task'

dataset = None
train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])

if args.dataset_type == 'torch':
    dataset = torchvision.datasets.Omniglot(os.getcwd(),
                                             download=True, transforms=train_transform)
else:
    dataset = OmniglotReactionTimeDataset(args.dataset_file, 
                transforms=train_transform)


validation_split = .2
shuffle_dataset = True

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    # np.random.seed(1)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                        sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=valid_sampler)

# convert to numpy array for classical ML techniques
if args.to_numpy: 
    dataset_features = next(iter(train_loader))[0].numpy()
    dataset_labels = next(iter(train_loader))[0].numpy()


