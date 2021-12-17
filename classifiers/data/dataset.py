"""
This file contains the class for the OmniglotReactionTimeDataset,
an in-house dataset prepared prior to this project.
"""

import pandas as pd

from PIL import Image, ImageFilter
from torch.utils.data import Dataset


class OmniglotReactionTimeDataset(Dataset):
    """
    Dataset for omniglot + reaction time data

    Dataset Structure:
    label1, label2, real_file, generated_file, reaction time
    ...

    args:
    - path: string - path to dataset (should be a csv file)
    - transforms: torchvision.transforms - transforms on the data
    """

    def __init__(self, data_file, transforms=None):
        self.raw_data = pd.read_csv(data_file)
        self.transform = transforms

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        label1 = int(self.raw_data.iloc[idx, 0])
        label2 = int(self.raw_data.iloc[idx, 1])
        im1name = self.raw_data.iloc[idx, 2]
        image1 = Image.open(im1name)
        im2name = self.raw_data.iloc[idx, 3]
        image2 = Image.open(im2name)

        rt = self.raw_data.iloc[idx, 4]
        sigma = self.raw_data.iloc[idx, 5]

        # just add in the blur for now, parameterize it later
        image1 = image1.filter(ImageFilter.GaussianBlur(radius=sigma))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        sample = {'label1': label1, 'label2': label2, 'image1': image1,
                  'image2': image2, 'rt': rt, 'acc': sigma}

        return sample
