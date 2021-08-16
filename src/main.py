print("\nrunning...\n")

import torch
from torch import nn
from torch import optim
from torch.tensor import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from datasets import nnwfDataset
from nnwf_models.nnwf01 import NNWF01

def trainLoop(
    dataloder:DataLoader, model:NNWF01, 
    optimizer:optim.Adam, lossFunc:nn.MSELoss
) -> float:
    model.train()
    for data, label in dataloder:
        pred = model(data)
        loss = lossFunc(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def testLoop(
    dataloader:DataLoader, model:NNWF01,
    lossFunc:nn.MSELoss, draw=False
) -> float:
    model.eval()
    countBatches = len(dataloader)
    loss = 0
    labelHist = []
    predHist = []
    with torch.no_grad():
        for data, label in dataloader:
            pred = model(data)
            loss += lossFunc(pred, label).item()
            if draw:
                predHist.extend(pred.tolist())
                labelHist.extend(label.tolist())

    loss /= countBatches

    if draw:
        return loss, labelHist, predHist
    else:
        return loss
    
epochs = 100
learningRate = 0.005

trainDataset, testDataset = nnwfDataset.readDataset2019_01(randomSeed=0) 
trainDataLoader = DataLoader(trainDataset,batch_size=64)
testDataLoader = DataLoader(testDataset,batch_size=64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = NNWF01().to(device)
optimizer = optim.Adam(model.parameters(), lr=learningRate)
lossFunc = nn.MSELoss()
print(model)


epochs = 500
trainLossHist = []
testLossHist = []
count = 1
for epoch in range(epochs):
    draw = False
    if epoch >= int(count * epochs / 10) or epoch == 0:
        draw = True
        count += 1
        print(f"epoch: {epoch}/{epochs}")
    trainLossHist.append(trainLoop(trainDataLoader, model, optimizer, lossFunc))
    if draw:
        testLoss, labelHist, predHist = testLoop(
            testDataLoader, model, lossFunc, draw=draw
        )
        testLossHist.append(testLoss)
    else:
        testLossHist.append(testLoop(
            testDataLoader, model, lossFunc, draw=draw)
        )

    if draw:        
        fig = plt.figure()
        plt.subplots_adjust(hspace=0.5)
        ytAx = fig.add_subplot(
            211, title="significant wave height", ylabel="significant wave height"
        )
        ypAx = fig.add_subplot(
            212, title="predict", ylabel="predict wave height"
        )
        print(sum(labelHist)/len(labelHist))
        ytAx.plot(range(len(labelHist)), labelHist)
        ypAx.plot(range(len(predHist)), predHist)
        plt.savefig(f"result/Yt_Yp{epoch}.jpg")        

fig = plt.figure()
plt.subplots_adjust(hspace=0.5)
trainAx = fig.add_subplot(
    211, title="train MSE Loss", ylabel="MSE loss", xlabel="epochs"
)
testAx = fig.add_subplot(
    212, title="test MSE Loss", ylabel="MSE loss", xlabel="epochs"
)
trainAx.plot(range(1,epochs+1), trainLossHist)
testAx.plot(range(1,epochs+1), testLossHist)
plt.savefig("result/loss.jpg")