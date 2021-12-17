"""
This file contains data preprocessor functions. In particular,
it contains just one related to the OmniglotReactionTimeDataset.
"""

from torch.utils.data import Dataset


def preprocess_reaction_time_data(data_source: Dataset):
    """
    This code is meant to be used with the OmniglotReactionTimeDataset class.
    In particular, it helps to adapt the output of that model so that the KFoldStratifiedSampler class
    can use it.
    Args:
        - data_source: an instance of the OmniglotReactionTimeDataset class.
    Returns: a list containing all pertinent objects for training and testing data from
    the OmniglotReactionTimeDataset class in the form of tuples consisting of data objects and corresponding labels.
    """
    return [(point["image1"], point["label1"]) for point in data_source]
