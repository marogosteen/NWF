import math

import matplotlib.pyplot as plt
import torch


class HistoryModel():
    def __init__(self) -> None:
        self.train_loss_hist = []
        self.eval_loss_list = []
        self.bestModelState = None

    def showResult(self):
        print("",
              "best epoch: ", f"\t{self.best_epoch()}",
              "best loss : ", f"\t{round(self.best_loss(), 7)}",
              "best RMSE : ", f"\t{round(math.sqrt(self.best_loss()), 7)}",
              sep="\n")

    def isBestLoss(self, currentLoss) -> bool:
        return self.best_loss() >= currentLoss or not self.bestModelState

    def best_loss(self):
        return min(self.eval_loss_list)

    def best_epoch(self):
        best_loss = self.best_loss()
        return self.eval_loss_list.index(best_loss) + 1

    def save_best_model_state(self, save_path):
        torch.save(self.bestModelState, save_path)

    def save_history(self, model_name):
        write_list = [
            f"best epoch {self.best_epoch()}\n",
            "best loss {:.6f}\n".format(self.best_loss()),
            "best standard deviation {:.3f}\n".format(math.sqrt(self.best_loss()))]
        with open(f"result/{model_name}/result_summary.txt", mode="w") as f:
            for line in write_list:
                f.write(line)

        self.draw_loss(model_name)

    def draw_loss(self, model_name):
        best_epoch = self.best_epoch()
        fig = plt.figure()
        ax = fig.add_subplot(
            111, title=f"loss  best epoch {best_epoch}",
            ylabel="MSE loss", xlabel="epochs")
        self.__plot_loss(ax, self.train_loss_hist, self.eval_loss_list)
        plt.savefig(f"result/{model_name}/loss.jpg")

    def __plot_loss(self, ax, train_loss, eval_loss):
        ax.plot(range(1, len(train_loss)+1), train_loss, label="train")
        ax.plot(range(1, len(eval_loss)+1), eval_loss, label="eval")
        ax.grid()
        ax.legend()
