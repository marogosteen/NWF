import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from models import nnwfDataset

trainDataset:Dataset = nnwfDataset.readDataset2019_01(randomSeed=0)
trainDataLoader = DataLoader(
    nnwfDataset.dataset2019_01(), batch_size=10
)

print(len(nnwfDataset.dataset2019_01(train=True)))
#trains = transforms.ToTensor(Dataset)