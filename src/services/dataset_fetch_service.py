import sqlite3

import torch


class Dataset_service():
    __inferiorityColumnIndex = 1

    def __init__(self, purpose:str) -> None:
        if not(purpose == "train" or purpose == "eval"):
            raise ValueError

        self.db = sqlite3.connect(database="database/dataset.db")
        self.__curosr = self.db.cursor()
        self.__query = open(f"src/services/query/get_{purpose}_dataset.sql").read()

    def select_record(self) -> None:
        self.__curosr.execute(self.__query)

    def full_record(self) -> list:
        self.select_record()
        return self.__curosr.fetchall()

    def next_buffer(self) -> list:
        return self.__curosr.fetchmany(5000)

    def truedata(self, forecast_hour:int) -> torch.Tensor:
        full_record_ndarray = torch.Tensor(self.full_record())
        shifted_before = full_record_ndarray[:-forecast_hour]
        shifted_after = full_record_ndarray[forecast_hour:]
        truedata = shifted_before[
            (shifted_before[:, self.__inferiorityColumnIndex] == 0) &
            (shifted_after[:, self.__inferiorityColumnIndex] == 0)
        ]
        return truedata