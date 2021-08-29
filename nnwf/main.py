print("\nrunning...\n")

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import NNWFDataset
from nets import NNWF_Net01


epochs = 40
learningRate = 0.005
batch_size = 64
modelName = "dataset01"


def trainLoop(
    dataloder:DataLoader, net, 
    optimizer:optim.Adam, lossFunc:nn.MSELoss) -> float:

    net.train()
    for data, label in dataloder:
        pred = net(data)
        loss = lossFunc(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def testLoop(
    dataloader:DataLoader, net,
    lossFunc:nn.MSELoss, draw=False) -> float:

    net.eval()
    countBatches = len(dataloader)
    loss = 0
    labelHist = []
    predHist = []
    with torch.no_grad():
        for data, label in dataloader:
            pred = net(data)
            loss += lossFunc(pred, label).item()
            if draw:
                labelHist.extend(label.tolist())
                predHist.extend(pred.tolist())
    loss /= countBatches

    if draw:
        return loss, labelHist, predHist
    else:
        return loss

def drawPredict(labelHist, predHist, epoch):
    fig = plt.figure()
    ax = fig.add_subplot(
        111, ylabel="wave height")
    ax.plot(range(len(labelHist)), labelHist, label="observed value")
    ax.plot(
        range(len(predHist)), predHist, 
        label="predicted value", alpha=0.5, color="red"
        )
    ax.grid()
    ax.legend()
    plt.savefig(f"result/Yt_Yp{epoch}.jpg")


trainDataset = NNWFDataset(modelName, mode="train")
testDataset = NNWFDataset(modelName, mode="eval")
trainDataLoader = DataLoader(trainDataset, batch_size=batch_size)
testDataLoader = DataLoader(testDataset, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(
    "Using {} device".format(device),
    "total data count:", 
    " train:", len(trainDataset), " test:", len(testDataset),
    "\n")

net = NNWF_Net01().to(device)
optimizer = optim.Adam(net.parameters(), lr=learningRate)
lossFunc = nn.MSELoss()
print(net)

trainLossHist = []
testLossHist = []
count = 0
for epoch in range(1, epochs+1):
    draw = False
    if epoch >= int(count * epochs / 10):
        draw = True
        count += 1
        print(f"epoch: {epoch}/{epochs}")

    trainLossHist.append(trainLoop(trainDataLoader, net, optimizer, lossFunc))
    if draw:
        testLoss, labelHist, predHist = testLoop(
            testDataLoader, net, lossFunc, draw=draw
        )
        testLossHist.append(testLoss)
    else:
        testLossHist.append(testLoop(
            testDataLoader, net, lossFunc, draw=draw)
        )

    if draw:        
        drawPredict(labelHist, predHist, epoch)

    if len(trainLossHist) - trainLossHist.index(min(trainLossHist)) > 5:
        print(f"\nOperate early stop epoch: {epoch}\n")
        break

trainDataLoader.dataset.db.close()
testDataLoader.dataset.db.close()

torch.save(net.state_dict(), f"nnwf/nets/state_dicts/{modelName}.pt")

fig = plt.figure()
ax = fig.add_subplot(
    111, ylabel="MSE loss", xlabel="epochs")
ax.plot(range(1, len(trainLossHist)+1), trainLossHist)
ax.plot(range(1, len(testLossHist)+1), testLossHist)
plt.savefig("result/loss.jpg")