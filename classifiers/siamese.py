import argparse
import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
import os
import pandas as pd
import random
# from skimage import io

from PIL import Image, ImageFilter

from torchvision.utils import save_image

class OmniglotReactionTimeDataset(Dataset):
    """
    Dataset for omniglot + reaction time data

    Dasaset Structure:
    label1, label2, real_file, generated_file, reaction time
    ...

    args:
    - path: string - path to dataset (should be a csv file)
    - transforms: torchvision.transforms - transforms on the data
    """

    def __init__(self, data_file, transforms=None):
        self.raw_data = pd.read_csv(data_file)
#         print('raw ', self.raw_data)
#        for i in range(5):
#            print('for testing purposes, the item is ', self.raw_data.iloc[0, i])
#            print('the type of the item is', type(self.raw_data.iloc[0, i]))

        self.transform = transforms

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        label1 = int(self.raw_data.iloc[idx, 0])
        label2 = int(self.raw_data.iloc[idx, 1])
        im1name = self.raw_data.iloc[idx, 2]
        image1 = Image.open(im1name)
        # save_image(image1, 'sample.png')
        im2name = self.raw_data.iloc[idx, 3]
        image2 = Image.open(im2name)
        
        rt = self.raw_data.iloc[idx, 4]
        sigma = self.raw_data.iloc[idx, 5]
        
        # just add in the blur for now, parameterize it later, 
        image1 = image1.filter(ImageFilter.GaussianBlur(radius = sigma))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        sample = {'label1': label1, 'label2': label2, 'image1': image1,
                                            'image2': image2, 'rt': rt, 'acc': sigma}

        return sample


class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
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
        #  return self.sigmoid(out)
        return out


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    # dataset = torchvision.datasets.Omniglot(os.getcwd(),
    #                                          download=True, transform=transform)

    # working with the 100-class dataset achieved better results
    # okay you are attempting to load in based upon the directory
    # you should be loading the csv file to deal with this, and handling stuff appropriately
    dataset = OmniglotReactionTimeDataset('sigma_dataset.csv', 
                transforms=transform)

    validation_split = .2
    shuffle_dataset = True

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        # np.random.seed(1)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                                    sampler=valid_sampler)

    #### Model work
    # now set to a Siamese network ...
    # model = torchvision.models.resnet50(pretrained=True)
    model = SiameseNetwork()


    # loss function and optimizer
    criterion = torch.abs_nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



    for epoch in range(2):
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
            outputs = model(image1,image2).to(device)
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