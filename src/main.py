print("\nrunning...\n")

import torch
from torch import nn
from torch import optim
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
    lossFunc:nn.MSELoss,
) -> float:
    model.eval()
    countBatches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for data, label in dataloader:
            pred = model(data)
            loss += lossFunc(pred, label).item()
    
    loss /= countBatches

    return loss
    
epochs = 100
learningRate = 0.05

trainDataset, testDataset = nnwfDataset.readDataset2019_01(randomSeed=0)
trainDataLoader = DataLoader(trainDataset,batch_size=64)
testDataLoader = DataLoader(testDataset,batch_size=64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = NNWF01().to(device)
optimizer = optim.Adam(model.parameters(), lr=learningRate)
lossFunc = nn.MSELoss()
print(model)


epochs = 1000
trainLossHist = []
testLossHist = []
count = 1
for epoch in range(epochs):
    if epoch >= int(count * epochs / 10):
        count += 1
        print(f"epoch: {epoch}/{epochs}")
    trainLossHist.append(trainLoop(trainDataLoader, model, optimizer, lossFunc))
    testLossHist.append(testLoop(testDataLoader, model, lossFunc))

fig = plt.figure()
trainAx = fig.add_subplot(
    211, title="train MSE Loss", ylabel="MSE loss", ylabel="epochs"
)
testAx = fig.add_subplot(
    212, title="test MSE Loss", ylabel="MSE loss", xlabel="epochs"
)
trainAx.plot(range(1,epochs+1), trainLossHist)
testAx.plot(range(1,epochs+1), testLossHist)