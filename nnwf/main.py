print("\nrunning...\n")

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import Train_NNWFDataset, Eval_NNWFDataset
from nets import NNWF_Net01


epochs = 100
learning_rate = 0.005
batch_size = 64
model_name = "nnwf01"


def main():
    train_loss_hist = []
    eval_loss_hist = []

    with Train_NNWFDataset() as train_dataset, Eval_NNWFDataset() as eval_dataset:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

        train_model = Train_model(train_dataloader, eval_dataloader)

        for epoch in range(1, epochs+1):
            draw_mode = True if epoch % int(epochs // 10) == 0 or epoch == 1 else False
            if draw_mode:
                print(f"epoch: {epoch}/{epochs}")
            
            train_loss_hist.append(train_model.train())
            eval_loss_hist.append(train_model.eval(epoch, draw_mode))

            if len(train_loss_hist) - train_loss_hist.index(min(train_loss_hist)) > 5:
                print(f"\nOperate early stop epoch: {epoch}\n")

        torch.save(train_model.net.state_dict(), f"nnwf/nets/state_dicts/{model_name}.pt")

class Train_model():
    def __init__(self, train_dataloader, eval_dataloader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.net =  NNWF_Net01().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.real_values = eval_dataloader.dataset.get_real_values()

    def train(self) -> float:
        net = self.net.train()
        for data, real_val in self.train_dataloader:
            pred = net(data)
            loss = self.loss_func(pred, real_val)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def eval(self, epoch:int, draw_mode:bool) -> float:
        net = self.net.eval()
        countBatches = len(self.eval_dataloader)
        loss = 0
        pred_hist = []
        with torch.no_grad():
            for data, real_val in self.eval_dataloader:
                pred = net(data)
                loss += self.loss_func(pred, real_val).item()
                if draw_mode:
                    pred_hist.extend(pred.tolist())
        loss /= countBatches

        if draw_mode:
            self.__draw_predict(pred_hist, epoch)

        return loss

    def __draw_predict(self, predHist, epoch):
        fig = plt.figure()
        plt.subplots_adjust(hspace=0.5)
        
        height_ax = fig.add_subplot(
            211, ylabel="wave height", title=f"epoch: {epoch}")
        height_ax.plot(range(len(self.real_values)), list(map(lambda x:x[0], self.real_values)), label="observed value")
        height_ax.plot(
            range(len(predHist)), list(map(lambda x:x[0], predHist)), 
            label="predicted value", alpha=0.5, color="red")
        height_ax.grid()
        height_ax.legend()

        period_ax = fig.add_subplot(
            212, ylabel="wave period", title=f"epoch: {epoch}")
        period_ax.plot(range(len(self.real_values)), list(map(lambda x:x[1], self.real_values)), label="observed value")
        period_ax.plot(
            range(len(predHist)), list(map(lambda x:x[1], predHist)),
            label="predicted value", alpha=0.5, color="red")
        period_ax.grid()
        period_ax.legend()
        
        plt.savefig(f"result/Yt_Yp{epoch}.jpg")

def draw_loss(train_loss_hist, eval_loss_hist):
    fig = plt.figure()
    ax = fig.add_subplot(
        111, ylabel="MSE loss", xlabel="epochs")
    ax.plot(range(1, len(train_loss_hist)+1), train_loss_hist, label="train")
    ax.plot(range(1, len(eval_loss_hist)+1), eval_loss_hist, label="eval")
    ax.grid()
    ax.legend()
    plt.savefig("result/loss.jpg")

if __name__ == "__main__":
    main()