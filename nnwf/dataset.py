import math

import numpy as np
import torch
from torch.utils.data import IterableDataset

from nnwf import orm as nnwf_orm


class DatasetBaseModel(IterableDataset):
    def __init__(
            self, service: nnwf_orm.DbService, forecastHour: int, trainHour: int):

        super().__init__()
        self.__dbService = service
        self.forecastHour = forecastHour
        self.trainHour = trainHour
        self.someRecords = []
        self.len, self.dataSize = self.__shape()

    def __shape(self):
        for count, (data, label) in enumerate(self):
            continue
        return count, len(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__dbService.close()

    def __len__(self):
        return self.len

    def __iter__(self):
        self.__dbService.initQuery()
        self.__dbService.fetchMany()
        return self

    def __next__(self):
        while True:
            if not self.someRecords:
                self.__fillSomeRows()
            self.someRecords.append(self.__dbService.nextRecord())

            if self.__includedInferiority():
                self.someRecords.pop(0)
                continue

            break

        if len(self.someRecords) != self.forecastHour + self.trainHour:
            exit()

        data = self.__traindataMolding()
        label = self.__labelMolding()
        self.someRecords.pop(0)

        return torch.Tensor(data), torch.Tensor(label)

    def __fillSomeRows(self) -> list:
        for _ in range(self.forecastHour + self.trainHour - 1):
            record = self.__dbService.nextRecord()
            self.someRecords.append(record)

    def __includedInferiority(self):
        for row in self.someRecords:
            for key in row.keys():
                if "_inferiority" in key:
                    if row[key]:
                        return True
        return False

    def __traindataMolding(self) -> list:
        data = []
        for row in self.someRecords[:self.trainHour]:
            normalize_month = row.datetime.month / 12
            normalize_hour = row.datetime.hour / 24
            sin_month = math.sin(normalize_month)
            cos_month = math.cos(normalize_month)
            sin_hour = math.sin(normalize_hour)
            cos_hour = math.cos(normalize_hour)
            isWindWave = False if row.period > row.height * 4 + 2 else True

            data.extend([
                sin_month,
                cos_month,
                sin_hour,
                cos_hour,

                isWindWave,
                row.air_pressure,
                row.temperature,

                row.kobe_latitude_velocity,
                row.kobe_longitude_velocity,
                row.kix_latitude_velocity,
                row.kix_longitude_velocity,
                row.tomogashima_latitude_velocity,
                row.tomogashima_longitude_velocity,
                row.akashi_latitude_velocity,
                row.akashi_longitude_velocity,
                row.osaka_latitude_velocity,
                row.osaka_longitude_velocity,

                row.height,
                row.period
            ])
        return data

    def __labelMolding(self) -> list:
        data = []
        forecast_index = self.trainHour + self.forecastHour - 1
        for row in self.someRecords[forecast_index: forecast_index + 1]:
            data.extend([
                row.height
            ])
        return data

    def close(self):
        self.__session.close()


class TrainDatasetModel(DatasetBaseModel):
    def __init__(
            self, forecast_hour: int, train_hour: int, targetyear: int):
        query = nnwf_orm.get_train_sqlresult(targetyear)
        service = nnwf_orm.DbService(query)
        super().__init__(
            service, forecast_hour, train_hour)

        self.mean = self.__getMean(self.len)
        self.std = self.__getStd(self.len, self.mean)

    def __getMean(self, len):
        mean = None
        for traindata, _ in self:
            traindata = traindata.numpy()
            if mean is None:
                mean = np.zeros(traindata.shape)
            mean += traindata / len
        return torch.FloatTensor(mean)

    def __getStd(self, len, mean):
        mean = mean.numpy()
        var = None
        for traindata, _ in self:
            traindata = traindata.numpy()
            if var is None:
                var = np.zeros(traindata.shape)
            var += np.sum(np.square(traindata - mean))/len
        std = np.sqrt(var)
        return torch.FloatTensor(std)


class EvalDatasetModel(DatasetBaseModel):
    def __init__(
            self, forecast_hour: int, train_hour: int, targetyear: int):
        query = nnwf_orm.getEvalSqlresult(targetyear)
        service = nnwf_orm.DbService(query)
        super().__init__(
            service, forecast_hour, train_hour)

    def getRealValues(self) -> torch.Tensor:
        return torch.stack([val for _, val in self], dim=0)


def classConvert(label):
    classTensor = torch.arange(0.25, 2.25, 0.25)
    index = torch.argmin(torch.abs(classTensor - label))
    return classTensor[index]


class TrainClassificationDataSetModel(TrainDatasetModel):
    def __init__(self, forecast_hour: int, train_hour: int, targetyear: int):
        super().__init__(forecast_hour, train_hour, targetyear)

    def __next__(self):
        data, label = super().__next__()
        label = classConvert(label)
        return data, label


class EvalClassificationDataSetModel(EvalDatasetModel):
    def __init__(self, forecast_hour: int, train_hour: int, targetyear: int):
        super().__init__(forecast_hour, train_hour, targetyear)

    def __next__(self):
        data, label = super().__next__()
        label = classConvert(label)
        return data, label
