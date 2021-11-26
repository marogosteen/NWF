import math

import numpy as np
from sqlalchemy import create_engine, orm
import torch
from torch.utils.data import IterableDataset

from nnwf import orm as nnwf_orm


class DatasetBaseModel(IterableDataset):
    def __init__(
            self, forecast_hour: int, train_hour: int, ormquery):

        super().__init__()
        engine = create_engine("sqlite:///database/dataset.db", echo=False)
        SessionClass = orm.sessionmaker(engine)

        self.__active_session: orm.session.Session = SessionClass()
        self.__query = ormquery
        self.forecast_hour = forecast_hour
        self.train_hour = train_hour
        self.len = self.__get_len()

    def __get_len(self):
        for count, _ in enumerate(self):
            continue
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__active_session.close()

    def __len__(self):
        return self.len

    def __iter__(self):
        self.__sqlresult = self.__active_session.execute(self.__query)
        self.__fetchmany()
        self.__rows = self.__fill_rows()
        return self

    def __fetchmany(self):
        list_buffer = self.__sqlresult.fetchmany(10000)
        if not list_buffer:
            raise StopIteration
        self.__iter_buffer = iter(list_buffer)

    def __fill_rows(self) -> list:
        rows = []
        for _ in range(self.forecast_hour + self.train_hour - 1):
            record = next(self.__iter_buffer)
            rows.append(record)
        return rows

    def __next__(self):
        while True:
            row = self.__buffer_next()
            self.__rows.append(row)
            is_inferiority = self.__inferiority_detector(self.__rows)
            if is_inferiority:
                self.__rows.pop(0)
            else:
                break
        data = self.__traindata_molding(self.__rows, self.train_hour)
        label = self.__label_molding(
            self.__rows, self.train_hour, self.forecast_hour)
        self.__rows.pop(0)
        return torch.Tensor(data), torch.Tensor(label)

    def __buffer_next(self):
        try:
            row = next(self.__iter_buffer)
        except StopIteration:
            self.__fetchmany()
            row = next(self.__iter_buffer)
        return row

    def __inferiority_detector(self, rows):
        for row in rows:
            inferiority = row.kobe_inferiority or row.kix_inferiority or\
                row.tomogashima_inferiority or row.nowphas_inferiority
            if inferiority:
                return True
        return False

    def __traindata_molding(self, rows, train_hour) -> list:
        data = []
        for row in rows[:train_hour]:
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
                row.kobe_latitude_velocity,
                row.kobe_longitude_velocity,
                row.kobe_temperature,
                row.kix_latitude_velocity,
                row.kix_longitude_velocity,
                row.kix_temperature,
                row.tomogashima_latitude_velocity,
                row.tomogashima_longitude_velocity,
                row.tomogashima_temperature,
                row.air_pressure,
                row.height,
                row.period
            ])
        return data

    def __label_molding(self, rows, train_hour, forecast_hour) -> list:
        data = []
        forecast_index = train_hour + forecast_hour - 1
        for row in rows[forecast_index: forecast_index + 1]:
            data.extend([
                row.height
            ])
        return data

    def close(self):
        self.__active_session.close()


class TrainDatasetModel(DatasetBaseModel):
    def __init__(
            self, forecast_hour: int, train_hour: int, targetyear: int):
        ormquery = nnwf_orm.get_train_sqlresult(targetyear)
        super().__init__(
            forecast_hour, train_hour, ormquery)
        self.mean = self.__get_mean(self.len)
        self.std = self.__get_std(self.len, self.mean)

    def __get_mean(self, len):
        mean = None
        for traindata, _ in self:
            traindata = traindata.numpy()
            if mean is None:
                mean = np.zeros(traindata.shape)
            mean += traindata / len
        return torch.FloatTensor(mean)

    def __get_std(self, len, mean):
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
        ormquery = nnwf_orm.get_eval_sqlresult(targetyear)
        super().__init__(
            forecast_hour, train_hour, ormquery)

    def get_real_values(self) -> torch.Tensor:
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
