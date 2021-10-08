import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from services import Dataset_service


class Train_NNWFDataset(IterableDataset):
    """
    TrainまたはEvalのどちらかの使用用途に合わせて、DBからデータを出力するSuperClass。
    withとforを用いて出力することを前提としている。
    """
    _inferiorityColumnIndex = 1
    _forecast_hour = 1
    _train_hour = 2

    def __init__(self, service: Dataset_service):
        """
        Base_NNWFDatasetのコンストラクタ。

        Args:
            service(Dataset_service) : DBからデータを読み込むService
        """
        super(Train_NNWFDataset).__init__()

        self._service = service
        self.len = len(service.truedata(self._forecast_hour))
        self._transform = self.__costom_transform()

    def __len__(self):
        return self.len

    def __iter__(self):
        self._service.select_record()
        self._iterable_buffer = iter(self._service.next_buffer())
        self._past_data = []

        for count in range(self._forecast_hour + self._train_hour - 1):
            record = torch.Tensor(next(self.__iterable_buffer))
            self._past_data.append(record)
        return self

    def __next__(self):
        while True:
            try:
                record = torch.FloatTensor(next(self.__iterable_buffer))
            except StopIteration:
                buffer = self._service.next_buffer()
                if len(buffer) == 0:
                    raise StopIteration

                self.__iterable_buffer = iter(buffer)
                record = torch.FloatTensor(next(self.__iterable_buffer))

            self._past_data.append(record)
            inferiority_count = list(
                map(lambda x: x[self._inferiorityColumnIndex], self._past_data)).count(1)
            if inferiority_count == 0:
                break
            else:
                self._past_data.pop(0)

        data = []
        for index in range(self._train_hour):
            data.append(self._transform(self._past_data[index][2:]))
        data = torch.cat(data, dim=0)
        ans = self._past_data[-1][-2:-1]
        self._past_data.pop(0)

        return data, ans

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._service.db.close()

    def __costom_transform(self) -> transforms.Lambda:
        train_service = Dataset_service("train")
        truedata: torch.Tensor = train_service.truedata(
            self._forecast_hour)[:, 2:]
        data_mean = truedata.mean(axis=0)
        data_std = truedata.std(axis=0)
        print(f"data\n\tmean:{data_mean}\n\tstd:{data_std}\n")
        return transforms.Lambda(lambda x: (x - data_mean) / data_std)


class Eval_NNWFDataset(Train_NNWFDataset):
    def __init__(
            self, service: Dataset_service):
        super(Eval_NNWFDataset, self).__init__(service)

    def get_real_values(self) -> torch.Tensor:
        return torch.stack([val for data, val in self], dim=0)
