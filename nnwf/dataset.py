import math
import sys
import datetime

import numpy as np
import torch
from torch.utils.data import IterableDataset

from nnwf.dbfetcher import DbFetcher, RecordModel


class DatasetBaseModel(IterableDataset):
    def __init__(
            self, fetcher: DbFetcher, forecastHour: int, trainHour: int):
        super().__init__()
        self.__fetcher = fetcher
        self.forecastHour = forecastHour
        self.trainHour = trainHour
        self.recordBuffer = []
        self.len, self.dataSize = self.__shape()

    def __shape(self):
        for count, (data, _) in enumerate(self):
            continue
        return count + 1, len(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__fetcher.close()

    def __len__(self):
        return self.len

    def __iter__(self):
        self.recordBuffer = []
        self.__fetcher.executeSql()
        return self

    def __next__(self):
        self.__loadBuffer()
        data = self.__traindataMolding()
        label = self.__labelMolding()
        return torch.Tensor(data), torch.Tensor(label)

    def __loadBuffer(self):
        """
        DBから1Recordを読み、Bufferにappendする。
        """
        while True:
            # BufferがEmpltyの場合、学習時間分のRecordを読む
            if not self.recordBuffer:
                self.__fillBuffer()
            else:
                self.recordBuffer.pop(0)
                self.recordBuffer.append(self.__fetcher.nextRecord())

            if self.__includedInferiority():
                continue

            break

        if len(self.recordBuffer) != self.trainHour + self.forecastHour:
            print("error ", file=sys.stderr)
            sys.exit(1)

    def __fillBuffer(self):
        """
        学習時間分のRecordを読み、recordBufferに格納する。
        """
        if self.recordBuffer:
            print("Error recordBufferが空ではない。", file=sys.stderr)

        for _ in range(self.forecastHour + self.trainHour):
            record = self.__fetcher.nextRecord()
            self.recordBuffer.append(record)

    def __includedInferiority(self):
        """
        recordBuffer内に不良Recordが含まれているかを返す
        """
        for record in self.recordBuffer:
            for v in record.__dict__.values():
                if v is None:
                    return True
        return False

    def __traindataMolding(self) -> list:
        """
        recordBufferに格納されたRecordを学習入力データに整形する。

        Returns:
        -----
            traindata(list): 学習用の整形された入力データ
        """
        data = []
        for index in range(self.trainHour):
            record: RecordModel = self.recordBuffer[index]
            normalize_month = record.datetime.month / 12
            normalize_hour = record.datetime.hour / 24
            sin_month = math.sin(2 * math.pi * normalize_month)
            cos_month = math.cos(2 * math.pi * normalize_month)
            sin_hour = math.sin(2 * math.pi * normalize_hour)
            cos_hour = math.cos(2 * math.pi * normalize_hour)
            isWindWave = int(False if record.period >
                             record.height * 4 + 2 else True)

            data.extend([
                sin_month,
                cos_month,
                sin_hour,
                cos_hour,

                isWindWave,
                record.air_pressure,
                record.temperature,

                record.kobe_velocity,
                record.kobe_sin_direction,
                record.kobe_cos_direction,
                record.kix_velocity,
                record.kix_sin_direction,
                record.kix_cos_direction,
                record.tomogashima_velocity,
                record.tomogashima_sin_direction,
                record.tomogashima_cos_direction,
                record.akashi_velocity,
                record.akashi_sin_direction,
                record.akashi_cos_direction,
                record.osaka_velocity,
                record.osaka_sin_direction,
                record.osaka_cos_direction,

                record.height,
                record.period
            ])

        return data

    def __labelMolding(self) -> list:
        """
        recordBufferに格納されたRecordを学習の真値データ（Label）に整形する。

        Returns:
        -----
            traindata(list): 学習用の整形された真値データ（Label）
        """
        data = []
        record = self.recordBuffer[self.trainHour + self.forecastHour - 1]
        data.extend([
            record.height
        ])
        return data

    def close(self):
        self.__session.close()

    def datetimeList(self) -> list:
        self.__fetcher.initQuery()
        datetimeList = []
        while True:
            try:
                datetimeList.append(
                    self.__fetcher.nextRecord().datetime.strftime("%Y-%m-%d %H:%M"))
            except StopIteration:
                break
        return datetimeList

    def inferiorityList(self) -> list:
        self.__iter__()
        inferiorityList = [True for _ in range(self.trainHour)]
        while True:
            try:
                if not self.recordBuffer:
                    self.__fillBuffer()
                else:
                    self.recordBuffer.pop(0)
                    self.recordBuffer.append(self.__fetcher.nextRecord())
                inferiorityList.append(self.__includedInferiority())
            except StopIteration:
                break
        return inferiorityList


class TrainDatasetModel(DatasetBaseModel):
    def __init__(
            self, forecast_hour: int, train_hour: int, targetyear: int):
        fetcher = DbFetcher(targetyear, mode="train")
        super().__init__(
            fetcher, forecast_hour, train_hour)

        self.mean = self.__getMean(self.len)
        self.std = self.__getStd(self.len, self.mean)

    def __getMean(self, len: int) -> torch.FloatTensor:
        """
        stream bufferを用いて入力データの平均値を求める。

        Args:
        -----
            len (int): 学習データのRecord数（1入力データあたりのSizeでは無い。）

        Returns:
        -----
            torch.FloatTensor: 要素数が1Recordの入力Sizeと等しい、それぞれのデータの平均。

        Example:
        ------
            self = torch.tensor([[1, 2, 3], [3, 3, 3], [5, 6, 7]])

            mean = __getMean( len(self) )

            print(mean)

            # tensor([3.0000, 3.6667, 4.3333])
        """

        mean = None
        for traindata, _ in self:
            traindata = traindata.numpy()
            if mean is None:
                mean = np.zeros(traindata.shape)
            mean += traindata / len
        return torch.FloatTensor(mean)

    def __getStd(self, len: int, mean: torch.FloatTensor):
        """
        stream bufferを用いて入力データの標準偏差を求める。

        Args:
        -----
            len (int): 学習データのRecord数（1入力データあたりのSizeでは無い。）

        Returns:
        -----
            torch.FloatTensor: 要素数が1Recordの入力Sizeと等しい、それぞれのデータの標準偏差。

        Example:
        -----
            self = torch.tensor([[1, 2, 3], [3, 3, 3], [5, 6, 7]])

            mean = __getMean( len(self) )

            std = __getStd(len(self), mean)

            print(std)

            # tensor([1.6330, 1.6997, 1.8856])
        """

        mean = mean.numpy()
        var = None
        for traindata, _ in self:
            traindata = traindata.numpy()
            if var is None:
                var = np.zeros(traindata.shape)
            var += np.square(traindata - mean)
        std = np.sqrt(var/len)
        return torch.FloatTensor(std)


class EvalDatasetModel(DatasetBaseModel):
    def __init__(
            self, forecast_hour: int, train_hour: int, targetyear: int):
        fetcher = DbFetcher(targetyear, mode="eval")
        super().__init__(
            fetcher, forecast_hour, train_hour)

    def observed(self) -> torch.Tensor:
        return torch.cat([val for _, val in self], dim=0)


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
