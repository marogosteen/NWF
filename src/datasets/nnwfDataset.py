import sqlite3

from torch.utils.data import IterableDataset

class NNWFDataset(IterableDataset):
    def __init__(self, mode="train"):
        assert mode == "train" or mode == "eval", "mode is train or eval"

        self.db = sqlite3.connect(database="database/dataset.db")        
        corsor = self.db.cursor()
        self._len = corsor.execute("select count(*) from dataset01").fetchone()[0]

        sql = f"select * from dataset01 where class = '{mode}'"
        self.tb = corsor.execute(sql)

    def __iter__(self):
        return self

    def __next__(self):
        row = self.tb.fetchone()
        if row == None:
            self.db.close()
            print("\n\nclose DB\n\n")
            raise StopIteration()
        
        data = row[2:-1]
        label = row[-1]
        return data, label

    def __len__(self):
        return self._len