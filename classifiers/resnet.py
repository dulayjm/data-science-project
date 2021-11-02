import argparse
import os
import numpy as np
import scipy.io
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import Dataset
import os
import pandas as pd
import random
# from skimage impoart io

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
    model = torchvision.models.resnet50(pretrained=True)


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

            # if args.loss_fn == 'psych-acc':
            #     psych = sample['acc']
            # else: 
            #     psych = sample['rt']

            # concatenate the batched images for now
            inputs = torch.cat([image1, image2], dim=0).to(device)
            labels = torch.cat([label1, label2], dim=0).to(device)

            # psych_tensor = torch.zeros(len(labels))
            # j = 0 
            # for i in range(len(psych_tensor)):
            #     if i % 2 == 0: 
            #         psych_tensor[i] = psych[j]
            #         j += 1
            #     else: 
            #         psych_tensor[i] = psych_tensor[i-1]
            # psych_tensor = psych_tensor.to(device)

            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)

            # if args.loss_fn == 'cross-entropy':
            #     loss = loss_fn(outputs, labels)
            # elif args.loss_fn == 'psych-acc': 
            #     loss = AccPsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)
            # else:
            #     loss = PsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()

            # labels_hat = torch.argmax(outputs, dim=1)  
            # correct += torch.sum(labels.data == labels_hat)

            # this seemed to fix the accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Finished Training')











    # dataiter = iter(train_loader)
    # sample = dataiter.next()
    # X = sample['image1']
    # y = sample['label1']

    # X = X.permute(1,2,3,0)

    # X = X.numpy()
    # y = y.numpy()

    # print('x shape', X.shape)
    # print('y.shape', y.shape)

    # dataiter = iter(validation_loader)
    # sample_test = dataiter.next()
    # X_test = sample_test['image1']
    # y_test = sample_test['label1']


    # X_test = X_test.permute(1,2,3,0)

    # X_test = X_test.numpy()
    # y_test = y_test.numpy()

    # X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
    # y = y.reshape(y.shape[0],)

    # X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1]*X_test.shape[2],X_test.shape[3]).T
    # y_test = y_test.reshape(y_test.shape[0],)

    # clf = \
    # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #        max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
    #        min_samples_split=2, min_weight_fraction_leaf=0.0,
    #        n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
    #        verbose=0, warm_start=False)

    # clf.fit(X, y)


    from sklearn.metrics import accuracy_score
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,preds))