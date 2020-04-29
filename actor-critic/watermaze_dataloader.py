"""
Pytorch data loader for the converted TFRecord data
"""
import torch
from torch.utils import data
from torch import Tensor
import pickle

class Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_loc="./data/new_watermaze_data_2.pkl", batch_size=128):
        "Initialization"
        self.data_loc = data_loc
        self.data = None
        self.batch_size = batch_size

    def __len__(self):
        "Denotes the total number of samples"
        return int(500)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data and get label
        infile = open(self.data_loc, 'rb')
        loaded_data = pickle.load(infile)
        index = index % len(loaded_data)

        X = (
            Tensor(loaded_data[index]["init_pos"]),
            Tensor(loaded_data[index]["init_hd"]),
            Tensor(loaded_data[index]["ego_vel"]),
        )
        y = (
            Tensor(loaded_data[index]["target_pos"]),
            Tensor(loaded_data[index]["target_hd"]).unsqueeze(dim=-1),
        )
        return X, y

