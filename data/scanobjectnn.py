import os
import math
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
#set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ScanObjectNN(Dataset):
    def __init__(self, data_path, split='train'):
        h5_file_path = os.path.join(data_path, 'training_objectdataset_augmentedrot.h5'
                                    if split in ['train', 'val'] else 'test_objectdataset_augmentedrot.h5')
        self.h5_file_path = h5_file_path
        self.data, self.labels = self.load_data()
        self.split = split
        if split in ['train', 'val']:
            self.split_data()

    def split_data(self):
        #shuffle data and labels in the same way
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.labels = self.labels[p]
        start, end = (0, int(len(self.data) * 0.8)) if self.split == 'train' else \
            (int(len(self.data) * 0.8), len(self.data))
        self.data = self.data[start:end]
        self.labels = self.labels[start:end]

    def load_data(self):
        with h5py.File(self.h5_file_path, 'r') as file:
            data = file['data'][:]
            labels = file['label'][:]

            # Read the datasets into arrays
            data_array = data[:]
            label_array = labels[:]
        return data_array, label_array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.normalize(self.data[idx]), self.labels[idx]

    def normalize(self, data):
        centroid = np.mean(data, axis=0)
        data -= centroid
        m = np.max(np.sqrt(np.sum(data**2, axis=1)))
        data /= m
        return data
