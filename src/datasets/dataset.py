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

    def __init__(self, service:Dataset_service):
        """
        Base_NNWFDatasetのコンストラクタ。

        Args:
            service(Dataset_service) : DBからデータを読み込むService
        """
        super(Train_NNWFDataset).__init__()

        self.__inferiorityColumnIndex = 1
        self.__forecast_hour = 1
        self.__service = service
        self.len = len(service.truedata(self.__forecast_hour))
        self.__transform = self.__costom_transform()

    def __len__(self):
        return self.len

    def __iter__(self): 
        self.__service.select_record()
        self.__iterable_buffer = iter(self.__service.next_buffer())
        self.__past_data = []

        for count in range(self.__forecast_hour):
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
            past_record = self.__past_data.pop(0)

            now_inferiority = record[self.__inferiorityColumnIndex]
            past_inferiority = past_record[self.__inferiorityColumnIndex]
            if now_inferiority == past_inferiority == 0:
                break

        data = self.__transform(past_record[2:])
        ans = record[-2:]
        return data, ans

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__service.db.close()

    # class __Custom_transform():
    #     def __init__(self, data:np.ndarray):
    #         self.mean = data.mean(axis=0).astype(np.float16)
    #         self.std = data.std(axis=0).astype(np.float16)
    #         print(f"data\n\tmean:{self.mean}\n\tstd:{self.std}\n")

    #     def normalize(self) -> transforms.Lambda:
    #         return transforms.Lambda(lambda x:(x - self.mean) / self.std)

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