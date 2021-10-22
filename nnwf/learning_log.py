import os
import math

import matplotlib.pyplot as plt
import torch


class LearningLog():
    train_loss_hist = []
    eval_loss_list = []
    best_model_state = None

    def best_loss(self):
        return min(self.eval_loss_list)

    def best_epoch(self):
        best_loss = self.best_loss()
        return self.eval_loss_list.index(best_loss) + 1

    def save_best_model_state(self, save_path):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.best_model_state, save_path)

    def save_log(self, model_name):
        write_list = [
            f"best epoch {self.best_epoch()}\n",
            "best loss {:.6f}\n".format(self.best_loss()),
            "best standard deviation {:.3f}\n".format(math.sqrt(self.best_loss()))]
        with open(f"result/{model_name}_result.txt", mode="w") as f:
            for line in write_list:
                f.write(line)

    def draw_loss(self, model_name):
        best_epoch = self.best_epoch()
        fig = plt.figure()
        ax = fig.add_subplot(
            111, title=f"loss  best epoch {best_epoch}",
            ylabel="MSE loss", xlabel="epochs")
        self.__plot_loss(ax, self.train_loss_hist, self.eval_loss_list)
        plt.savefig(f"result/{model_name}_loss.jpg")

    def __plot_loss(self, ax, train_loss, eval_loss):
        ax.plot(range(1, len(train_loss)+1), train_loss, label="train")
        ax.plot(range(1, len(eval_loss)+1), eval_loss, label="eval")
        ax.grid()
        ax.legend()
