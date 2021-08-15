import torch
from torch.utils.data import Dataset
from torchvision import transforms

trains = transforms.ToTensor(Dataset)
class NNWFDataset2019(Dataset):
    def __init__(self) -> None:
        dataDir = "data/dataset2019/"
        amedasData = self.__readAmedas(dataDir+"amedas2019.csv")
        self.data = "data/Dataset2019"
        self.label = "data/Dataset2019"

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data:torch.Tensor = self.data[index]
        label:torch.Tensor = self.label[index]
        return data, label

    def __readAmedas(readFilePath:str) -> torch.Tensor:
        data:list = []
        with open(readFilePath, encoding="Shift-jis", mode="r") as f:
            for line in f.readlines()[6:]:
                line = line.strip().split(",")
                line[0]
                

        return torch.FloatTensor(data)