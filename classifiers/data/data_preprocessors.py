from torch.utils.data import Dataset


def preprocess_reaction_time_data(data_source: Dataset):
    return [(point["image1"], point["label1"]) for point in data_source]
