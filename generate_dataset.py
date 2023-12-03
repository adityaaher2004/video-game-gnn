from typing import List, Tuple, Union
import torch
from torch_geometric.data import Dataset, Data
import numpy as np

class GamesDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root -> root directory of data
        raw_dir -> downloaded dataset
        processed_dir -> processed data
        """
        super(GamesDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        if file exists, download is not triggered
        """
        return "games.csv"
    
    @property
    def processed_file_names(self):
        """
        process skipped if file not found
        """
        return "not_implemented.pt"
    
    def download(self):
        pass

    def process(self):
        pass