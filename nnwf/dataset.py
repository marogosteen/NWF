import math

import numpy as np
from sqlalchemy import create_engine, orm
import torch
from torch.utils.data import IterableDataset

from nnwf import orm as nnwf_orm


class DatasetBaseModel(IterableDataset):
    def __init__(
            self, forecastHour: int, trainHour: int, query):

        super().__init__()
        engine = create_engine("sqlite:///database/dataset.db", echo=False)
        SessionClass = orm.sessionmaker(engine)

        self.__session: orm.session.Session = SessionClass()
        self.__query = query
        self.forecastHour = forecastHour
        self.trainHour = trainHour
        self.len = self.__countData()

    def __countData(self):
        for count, _ in enumerate(self):
            continue
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__session.close()

    def __len__(self):
        return self.len

    def __iter__(self):
        self.__sqlResult = self.__session.execute(self.__query)
        self.__fetchSqlResult()
        self.__someRows = self.__fillSomeRows()
        return self

    def __fetchSqlResult(self):
        buffer = self.__sqlResult.fetchmany(10000)
        if not buffer:
            raise StopIteration
        self.__iterBuffer = iter(buffer)

    def __fillSomeRows(self) -> list:
        rows = []
        for _ in range(self.forecastHour + self.trainHour - 1):
            record = next(self.__iterBuffer)
            rows.append(record)
        return rows

    def __next__(self):
        self.__reloadSomeRows()
        data = self.__traindataMolding()
        label = self.__labelMolding()
        self.__someRows.pop(0)
        return torch.Tensor(data), torch.Tensor(label)

    def __reloadSomeRows(self):
        while True:
            row = self.__bufferNext()
            self.__someRows.append(row)
            if self.__isInferiority():
                self.__someRows.pop(0)
            else:
                break

    def __bufferNext(self):
        try:
            row = next(self.__iterBuffer)
        except StopIteration:
            self.__fetchSqlResult()
            row = next(self.__iterBuffer)
        return row

    def __isInferiority(self):
        for row in self.__someRows:
            for key in row.keys():
                if "_inferiority" in key:
                    if row[key]:
                        return True
        return False

    def __traindataMolding(self) -> list:
        data = []
        for row in self.__someRows[:self.trainHour]:
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
        for row in self.__someRows[forecast_index: forecast_index + 1]:
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
        super().__init__(
            forecast_hour, train_hour, query)
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
        super().__init__(
            forecast_hour, train_hour, query)

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
