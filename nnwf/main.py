print("\nrunning...\n")

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import NNWFDataset
from nets import NNWF_Net01


epochs = 100
learning_rate = 0.005
batch_size = 64
model_name = "nnwf01"

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
        111, ylabel="wave height", title=f"epoch: {epoch}")
    ax.plot(range(len(labelHist)), labelHist, label="observed value")
    ax.plot(
        range(len(predHist)), predHist, 
        label="predicted value", alpha=0.5, color="red")
    ax.grid()
    ax.legend()

    # height_ax = fig.add_subplot(
    #     211, ylabel="wave height", title=f"epoch: {epoch}")
    # height_ax.plot(range(len(labelHist)), list(map(lambda x:x[0], labelHist)), label="observed value")
    # height_ax.plot(
    #     range(len(predHist)), list(map(lambda x:x[0], predHist)), 
    #     label="predicted value", alpha=0.5, color="red")
    # height_ax.grid()
    # height_ax.legend()

    # period_ax = fig.add_subplot(
    #     212, ylabel="wave period", title=f"epoch: {epoch}")
    # period_ax.plot(range(len(labelHist)), list(map(lambda x:x[1], labelHist)), label="observed value")
    # period_ax.plot(
    #     range(len(predHist)), list(map(lambda x:x[1], predHist)),
    #     label="predicted value", alpha=0.5, color="red")
    # period_ax.grid()
    # period_ax.legend()
    
    plt.savefig(f"result/Yt_Yp{epoch}.jpg")


train_dataset = NNWFDataset(mode="train")
eval_dataset = NNWFDataset(mode="eval")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(
    "Using {} device\n".format(device),
    "total data count\n", 
    " train:", len(train_dataset), " test:", len(eval_dataset),
    "\n")

net = NNWF_Net01().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()
print(net)

trainLossHist = []
testLossHist = []

for epoch in range(1, epochs+1):
    draw = False
    if epoch % int(epochs // 10) == 0 or epoch == 1:
        draw = True
        print(f"epoch: {epoch}/{epochs}")

    trainLossHist.append(trainLoop(train_dataloader, net, optimizer, loss_func))
    if draw:
        testLoss, labelHist, predHist = testLoop(
            eval_dataloader, net, loss_func, draw=draw
        )
        testLossHist.append(testLoss)
    else:
        testLossHist.append(testLoop(
            eval_dataloader, net, loss_func, draw=draw)
        )

    if draw:        
        drawPredict(labelHist, predHist, epoch)

    if len(trainLossHist) - trainLossHist.index(min(trainLossHist)) > 5:
        print(f"\nOperate early stop epoch: {epoch}\n")
        #break

train_dataloader.dataset.db.close()
eval_dataloader.dataset.db.close()

torch.save(net.state_dict(), f"nnwf/nets/state_dicts/{model_name}.pt")

fig = plt.figure()
heightax = fig.add_subplot(
    111, ylabel="MSE loss", xlabel="epochs")
heightax.plot(range(1, len(trainLossHist)+1), trainLossHist, label="train")
heightax.plot(range(1, len(testLossHist)+1), testLossHist, label="eval")
heightax.grid()
heightax.legend()
plt.savefig("result/loss.jpg")