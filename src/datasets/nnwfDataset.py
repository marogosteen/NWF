import random

import torch
from torch.utils.data import Dataset

def readDataset2019_01(trainPercentage:float=0.7 ,randomSeed=None):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self,index):
            return self.data[index][:-1], self.data[index][-1]

    message:str = f"trainPercentage({trainPercentage})を0以上1以下にしてください．"
    if trainPercentage > 1.0:
        raise Exception(message)
    elif trainPercentage < 0.0:
        raise Exception(message)

    if randomSeed == None:
        random.seed()
    else:
        random.seed(randomSeed)

    dataPath:str = "dataset/2019_01.csv"
    data:list = []
    with open(dataPath) as f:
        for line in f.readlines():
            line = list(map(lambda x:float(x),line.strip().split(",")))
            data.append(line)
    random.shuffle(data)
    data:torch.FloatTensor = torch.FloatTensor(data)

    trainDataLength:int = int(round(len(data) * trainPercentage,0))
    trainData:torch.FloatTensor = data[:trainDataLength]
    testData:torch.FloatTensor = data[trainDataLength:]

    trainDataset = MyDataset(trainData)
    testDataset = MyDataset(testData)

    return trainDataset, testDataset