import numpy as np
import torch

from nwf.datasets.dbfetcher import DbFetcher
from nwf.datasets.basedataset import DatasetBaseModel


class TrainDatasetModel(DatasetBaseModel):
    def __init__(
            self, forecastHour: int, trainHour: int, fetcher: DbFetcher):
        super().__init__(
            fetcher, forecastHour, trainHour)

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
            self, forecastHour: int, trainHour: int, fetcher: DbFetcher):
        super().__init__(
            fetcher, forecastHour, trainHour)

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
