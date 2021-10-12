import argparse
import os
import numpy as np
import scipy.io
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.Omniglot(os.getcwd(),
                                             download=True, transform=transform)

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

    dataiter = iter(train_loader)
    X, y = dataiter.next()

    X = X.permute(1,2,3,0)

    X = X.numpy()
    y = y.numpy()

    print('x shape', X.shape)
    print('y.shape', y.shape)

    dataiter = iter(validation_loader)
    X_test, y_test = dataiter.next()

    X_test = X_test.permute(1,2,3,0)

    X_test = X_test.numpy()
    y_test = y_test.numpy()

    X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
    y = y.reshape(y.shape[0],)

    X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1]*X_test.shape[2],X_test.shape[3]).T
    y_test = y_test.reshape(y_test.shape[0],)

    clf = \
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

    clf.fit(X, y)


    from sklearn.metrics import accuracy_score
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,preds))