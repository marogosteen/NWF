import random

import torch
from torch.utils.data import Dataset

def readDataset2019_01(self, trainPercentage:float=0.7 ,randomSeed=None):

    class MyDataset(Dataset):
        def __init__(self, dataPath:str, dataIndex:list):
            self.dataPath:str = dataPath
            self.dataIndex:list = dataIndex

        def __len__(self):
            return len(self.dataIndex)
        
        def __getitem__(self,index):
            index:int = self.dataIndex[index]
            with open(self.dataPath) as f:
                line = f.readlines()[index].strip().split(",")
                data = torch.FloatTensor(list(map(lambda x:float(x), line)))
                label = data[-1]
            return data[:-1], label

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
    with open(self.dataPath) as f:
        datalength:int = len(f.readlines())
    fullDataIndex:list = list(range(datalength))
    trainIndex:list = random.sample(
        fullDataIndex,int(round(self.datalength * trainPercentage,0))
    )
    testIndex:list = list(set(fullDataIndex) - set(trainIndex))

    trainDataset = MyDataset(dataPath,trainIndex)
    testDataset = MyDataset(dataPath,testIndex)

    return trainDataset, testDataset