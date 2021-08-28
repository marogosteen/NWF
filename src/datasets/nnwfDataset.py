import sqlite3

import torch
from torch.utils.data import IterableDataset


class NNWFDataset(IterableDataset):
    def __init__(self, mode="train"):
        super().__init__()
        assert mode == "train" or mode == "eval", "mode is train or eval"
        self.mode = mode

        self.db = sqlite3.connect(database="database/dataset.db")     
        self.len = self.db.cursor()\
            .execute(f"select count(*) from dataset01 where class = '{mode}'")\
            .fetchone()[0]

    def __iter__(self):
        self.tb = self.db.cursor().execute(f"select * from dataset01 where class = '{self.mode}'")

        self.iter_counter = 0

        return self

    def __next__(self):
        if self.iter_counter == self.len:
            raise StopIteration

        row = self.tb.fetchone()
        row = torch.FloatTensor(row[2:])
        data = row[:-1]
        label = row[-1:]

        self.iter_counter += 1
        return data, label

    def __len__(self):
        return self.len