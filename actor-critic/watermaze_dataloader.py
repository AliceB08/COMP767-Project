"""
Pytorch data loader for the converted TFRecord data
"""
import torch
from torch.utils import data
from torch import Tensor
import glob
import pickle

class Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_loc="./data/watermaze_data.pkl", batch_size=256):
        "Initialization"
        self.data_loc = data_loc
        self.data = None
        self.batch_size = batch_size

    def __len__(self):
        "Denotes the total number of samples"
        return int(1e6)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data and get label
        index = index % 10000
        infile = open(self.data_loc, 'rb')
        loaded_data = pickle.load(infile)
        #loaded_data = self.loaded_data()
        X = (
            Tensor(loaded_data["init_pos"][index]),
            Tensor(loaded_data["init_hd"][index]),
            Tensor(loaded_data["ego_vel"][index]),
        )

        y = (
            Tensor(loaded_data["target_pos"][index]),
            Tensor(loaded_data["target_hd"][index]),
        )
        return X, y

    # def loaded_data(self):
    #     if self.data == None or self.j % self.batch_size == 0:
    #         file_id = torch.randint(high=len(self.file_list), size=(1,))
    #         ID = self.file_list[file_id]
    #         self.data = torch.load(ID)
    #
    #     self.j += 1
    #     return self.data
