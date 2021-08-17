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
            return self.data[index][:4], self.data[index][4]

    trainPercentageErrorDetection(trainPercentage)

    if randomSeed == None:
        random.seed()
    else:
        random.seed(randomSeed)

    dataPath:str = "dataset/2019_01.csv"
    data:list = loadData(dataPath)
    random.shuffle(data)
    data:torch.FloatTensor = torch.FloatTensor(data)

    bias = 3.5
    grad = 4 / 2
    data = selectData(data, grad, bias)

    trainDataCount:int = int(round(len(data) * trainPercentage,0))
    trainData:torch.FloatTensor = data[:trainDataCount]
    testData:torch.FloatTensor = data[trainDataCount:]

    trainDataset = MyDataset(trainData)
    testDataset = MyDataset(testData)

    return trainDataset, testDataset

def trainPercentageErrorDetection(trainPercentage):
    message:str = f"trainPercentage({trainPercentage})を0以上1以下にしてください．"
    if trainPercentage > 1.0:
        raise Exception(message)
    elif trainPercentage < 0.0:
        raise Exception(message)

def loadData(dataPath):
    data:list = []
    with open(dataPath) as f:
        for line in f.readlines():
            line = list(map(lambda x:float(x),line.strip().split(",")))
            data.append(line)
    return data

def selectData(data, grad, bias):
    heightIndex = 4
    periodIndex = 5
    data = data[data[:,periodIndex] < data[:,heightIndex] * grad + bias]
    return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    selectedData, _ = readDataset2019_01(trainPercentage=1.0)
    selectedData = selectedData.data
    print("selected data shape:", selectedData.shape)

    fig = plt.figure()
    ax = fig.add_subplot(
        ylabel="period [sec]", xlabel="height [m]",
        ylim=(1.5, 10), xlim=(0, 2.0)
    )
    ax.scatter(selectedData[:,4], selectedData[:,5])
    
    plt.show()