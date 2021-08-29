import sqlite3

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms


class NNWFDataset(IterableDataset):
    def __init__(self, datasetName:str, mode="train"):
        super().__init__()
        assert mode == "train" or mode == "eval", "mode is train or eval"
        self.datasetName = datasetName
        self.mode = mode

        self.db = sqlite3.connect(database="database/dataset.db")     
        fullData = np.array(self.db.cursor()\
            .execute(f"select * from {datasetName} where class = '{mode}'")\
            .fetchall())[:, 2:-1].astype(np.float32)
        self.len = len(fullData)
        self.normalize = Custom_transform(fullData).normalize()

    def __iter__(self):
        self.tb = self.db.cursor().execute(f"select * from {self.datasetName} where class = '{self.mode}'")
        self.iterCounter = 0

        return self

    def __next__(self):
        if self.iterCounter == self.len:
            raise StopIteration

        row = self.tb.fetchone()
        row = torch.FloatTensor(row[2:])
        data = self.normalize(row[:-1])
        label = row[-1:]

        self.iterCounter += 1
        return data, label

    def __len__(self):
        return self.len


class Custom_transform():
    def __init__(self, data:np.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
    
    def normalize(self):
        return transforms.Lambda(lambda x:(x-self.mean)/self.std)