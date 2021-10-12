import math

from sqlalchemy import create_engine, orm
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from nnwf.datasets import orm as nnwf_orm


class DatasetBaseModel(IterableDataset):
    def __init__(self, forecast_hour: int, train_hour: int):
        super(DatasetBaseModel).__init__()
        engine = create_engine("sqlite:///database/dataset.db", echo=True)
        SessionClass = orm.sessionmaker(engine)
        self._active_session: orm.session.Session = SessionClass()
        self._sqlresult = nnwf_orm.get_sqlresult(self._active_session)
        self.forecast_hour = forecast_hour
        self.train_hour = train_hour
        self.len
        self.mean
        self.std

    def __datainfo(self):
        while True:
            exit()
        for count, [data, label] in enumerate(self):
            continue
        self.len = count
        self.mean = sum_val / count
        self.std = hoge

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._active_session.close()

    def __len__(self):
        return self.len

    def __iter__(self):
        list_buffer = self._sqlresult.fetchmany(1000)
        if not list_buffer:
            raise StopIteration
        self._iter_buffer = list_buffer
        self._rows = self.__fill_rows()
        return self

    def __fill_rows(self) -> list:
        rows = []
        for count in range(self.forecast_hour + self.train_hour - 1):
            record = next(self._iter_buffer)
            rows.append(record)
        return rows

    def __next__(self):
        is_inferiority = False
        while not is_inferiority:
            row = self.__buffer_next()
            self._rows.append(row)
            is_inferiority = self.__inferiority_detector(self._rows)
            if is_inferiority:
                self._rows.pop(0)
        data = self.__traindata_molding(self._rows)
        label = self.__label_molding(self._rows)
        return torch.Tensor(data), torch.Tensor(label)

    def __buffer_next(self):
        try:
            row = next(self._iter_buffer)
        except StopIteration:
            self.__iter__()
            row = next(self._iter_buffer)
        return row

    def __inferiority_detector(self, rows):
        for row in rows:
            is_inferiority = \
                False == row.kobe.inferiority == row.kix.inferiority == \
                row.tomogashima.inferiority == row.nowphas.inferiority
            if is_inferiority:
                return True
        return False

    def __traindata_molding(self, rows) -> list:
        data = []
        for row in rows[:self.train_hour]:
            sin_datetime = math.sin(row.kobe.datetime)
            cos_datetime = math.cos(row.kobe.dateitme)
            data.extend([
                sin_datetime,
                cos_datetime,
                row.kobe.latitude_velocity,
                row.kobe.longitude_velocity,
                row.kobe.temperature_velocity,
                row.kix.latitude_velocity,
                row.kix.longitude_velocity,
                row.kix.temperature_velocity,
                row.tomogashima.latitude_velocity,
                row.tomogashima.longitude_velocity,
                row.tomogashima.temperature_velocity,
                row.nowphas.significant_height,
                row.nowphas.significant_period,
                row.nowphas.direction
            ])
        return data

    def __label_molding(self, rows) -> list:
        data = []
        forecast_index = self.train_hour + self.forecast_hour - 1
        for row in rows[forecast_index: forecast_index + 1]:
            data.extend([
                row.nowphas.significant_height,
            ])
        return data

    def close(self):
        self._active_session.close()


class Train_NNWFDataset(DatasetBaseModel):
    def __init__(self, forecast_hour: int, train_hour: int):
        super().__init__(forecast_hour, train_hour)


class Eval_NNWFDataset(DatasetBaseModel):
    def __init__(self, forecast_hour: int, train_hour: int):
        super().__init__(forecast_hour, train_hour)

    def get_real_values(self) -> torch.Tensor:
        return torch.stack([val for data, val in self], dim=0)
