import sqlite3

import torch
from torch.utils.data import IterableDataset


class NNWFDataset(IterableDataset):
    def __init__(self, datasetName:str, mode="train"):
        super().__init__()
        assert mode == "train" or mode == "eval", "mode is train or eval"
        self.datasetName = datasetName
        self.mode = mode

        self.db = sqlite3.connect(database="database/dataset.db")     
        self.len = self.db.cursor()\
            .execute(f"select count(*) from {datasetName} where class = '{mode}'")\
            .fetchone()[0]

    def __iter__(self):
        self.tb = self.db.cursor().execute(f"select * from {self.datasetName} where class = '{self.mode}'")
        self.iterCounter = 0

        return self

    def __next__(self):
        if self.iterCounter == self.len:
            raise StopIteration

        row = self.tb.fetchone()
        row = torch.FloatTensor(row[2:])
        data = row[:-1]
        label = row[-1:]

        self.iterCounter += 1
        return data, label

    def __len__(self):
        return self.len