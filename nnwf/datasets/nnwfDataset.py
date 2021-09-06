import sqlite3

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms


class NNWFDataset(IterableDataset):
    def __init__(self, mode=None):
        super().__init__()
        assert mode == "train" or mode == "eval", "mode is train or eval"

        self.db = sqlite3.connect(database="database/dataset.db")
        self.__query = open(f"database/query/get_{mode}_dataset.sql").read()

        ndarray_full_record = self.__get_ndarray_record()
        self.len = len(ndarray_full_record)
        self.__normalize = Custom_transform(ndarray_full_record).normalize()

        self.__past_data = None

    def __get_selected_record(self):
        return self.db.cursor().execute(self.__query)
    
    def __get_record_iterator(self, selected_record):
        buffer = selected_record.fetchmany(10000)
        if len(buffer) == 0:
            raise StopIteration
        iterable_buffer = iter(buffer)
        return iterable_buffer

    def __get_ndarray_record(self):
        selected_record = self.__get_selected_record().fetchall()
        ndarrary_record = np.array(selected_record, dtype=np.float32)
        return ndarrary_record

    def __iter__(self): 
        self.__selected_record = self.__get_selected_record()
        one_record = self.__selected_record.fetchone()
        self.__past_data = torch.Tensor(one_record)
        self.__iterable_buffer = self.__get_record_iterator(self.__selected_record)
        return self

    def __next__(self):
        try:
            record = torch.FloatTensor(next(self.__iterable_buffer))
        except StopIteration:
            self.__iterable_buffer = self.__get_record_iterator(self.__selected_record)
            record = torch.FloatTensor(next(self.__iterable_buffer))

        data = self.__normalize(self.__past_data)
        label = record[-2:]
        self.__past_data = record
        return data, label

    def __len__(self):
        return self.len

class Custom_transform():
    def __init__(self, data:np.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def normalize(self):
        return transforms.Lambda(lambda x:(x - self.mean) / self.std)