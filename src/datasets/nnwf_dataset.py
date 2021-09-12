import sqlite3

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms


class Base_NNWFDataset(IterableDataset):
    def __init__(self, mode=None):
        super(Base_NNWFDataset).__init__()
        assert mode == "train" or mode == "eval", "mode is train or eval"
        self.__difference_time = 1

        self.db = sqlite3.connect(database="database/dataset.db")
        self.__query = open(f"database/query/get_{mode}_dataset.sql").read()

        ndarray_full_record = self.__ndarray_full_record()
        self.len = len(ndarray_full_record)
        self.__normalize = self.__Custom_transform(ndarray_full_record).normalize()

    def __len__(self):
        return self.len - self.__difference_time

    def __iter__(self): 
        self.__selected_record = self.__select_record()
        self.__past_tensor_list = self.__fetch_past_tensor_list()
        self.__iterable_buffer = self.__iteratable_buffer()
        return self

    def __next__(self):
        try:
            record = torch.FloatTensor(next(self.__iterable_buffer))
        except StopIteration:
            self.__iterable_buffer = self.__iteratable_buffer()
            record = torch.FloatTensor(next(self.__iterable_buffer))

        data = self.__normalize(self.__past_tensor_list.pop(0))
        label = record[-2:]
        self.__past_tensor_list.append(record)
        return data, label

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.close()

    def __select_record(self):
        return self.db.cursor().execute(self.__query)

    def __fetch_past_tensor_list(self):
        return list(map(
            lambda record:torch.FloatTensor(record),  
            self.__selected_record.fetchmany(self.__difference_time)
        ))

    def __iteratable_buffer(self):
        buffer = self.__selected_record.fetchmany(10000)
        if len(buffer) == 0:
            raise StopIteration
        iterable_buffer = iter(buffer)
        return iterable_buffer

    def __ndarray_full_record(self):
        selected_record = self.__select_record().fetchall()
        ndarrary_record = np.array(selected_record, dtype=np.float32)
        return ndarrary_record

    class __Custom_transform():
        def __init__(self, data:np.ndarray):
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)

        def normalize(self):
            return transforms.Lambda(lambda x:(x - self.mean) / self.std)

class Train_NNWFDataset(Base_NNWFDataset):
    def __init__(self):
        super(Train_NNWFDataset, self).__init__("train")

class Eval_NNWFDataset(Base_NNWFDataset):
    def __init__(self):
        super(Eval_NNWFDataset, self).__init__("eval")
    
    def get_real_values(self):
        return torch.stack([val for data, val in self], dim=0)