import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from services import Dataset_service


class Train_NNWFDataset(IterableDataset):
    """
    TrainまたはEvalのどちらかの使用用途に合わせて、DBからデータを出力するSuperClass。
    withとforを用いて出力することを前提としている。
    """
    __inferiorityColumnIndex = 1
    __forecast_hour = 1
    # 何時間分の学習なのか
    __foo_hour = 2

    def __init__(self, service:Dataset_service):
        """
        Base_NNWFDatasetのコンストラクタ。

        Args:
            service(Dataset_service) : DBからデータを読み込むService
        """
        super(Train_NNWFDataset).__init__()

        self.__service = service
        self.len = len(service.truedata(self.__forecast_hour))
        self.__transform = self.__costom_transform()

    def __len__(self):
        return self.len

    def __iter__(self): 
        self.__service.select_record()
        self.__iterable_buffer = iter(self.__service.next_buffer())
        self.__past_data = []

        for count in range(self.__forecast_hour + self.__foo_hour - 1):
            record = torch.Tensor(next(self.__iterable_buffer))
            self.__past_data.append(record)
        return self

    def __next__(self):
        while True:
            try:
                record = torch.FloatTensor(next(self.__iterable_buffer))
            except StopIteration:
                buffer = self.__service.next_buffer()
                if len(buffer) == 0:
                    raise StopIteration
                
                self.__iterable_buffer = iter(buffer)
                record = torch.FloatTensor(next(self.__iterable_buffer))

            self.__past_data.append(record)
            inferiority_count = list(map(lambda x: x[self.__inferiorityColumnIndex], self.__past_data)).count(1)
            if inferiority_count == 0:
                break
            else:
                self.__past_data.pop()
            # past_record = self.__past_data.pop(0)
            # self.__foo.append(past_record)

            # now_inferiority = record[self.__inferiorityColumnIndex]
            # past_inferiority = past_record[self.__inferiorityColumnIndex]
            # if now_inferiority == past_inferiority == 0:
            #     break

        data = []
        for index in range(self.__foo_hour):
            data.append(self.__transform(self.__past_data[index][2:]))
        data = torch.cat(data, dim=0)
        ans = self.__past_data[-1][-2:]
        self.__past_data.pop(0)

        # data = self.__transform(past_record[2:])
        # ans = record[-2:]
        return data, ans

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__service.db.close()

    def __costom_transform(self) -> transforms.Lambda:
        train_service = Dataset_service("train")
        truedata:torch.Tensor = train_service.truedata(self.__forecast_hour)[:, 2:]
        data_mean = truedata.mean(axis=0)
        data_std = truedata.std(axis=0)
        print(f"data\n\tmean:{data_mean}\n\tstd:{data_std}\n")
        return transforms.Lambda(lambda x:(x - data_mean) / data_std)


class Eval_NNWFDataset(Train_NNWFDataset):
    def __init__(
        self, service:Dataset_service):
        super(Eval_NNWFDataset, self).__init__(service)
    
    def get_real_values(self) -> torch.Tensor:
        return torch.stack([val for data, val in self], dim=0)