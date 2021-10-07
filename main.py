import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from services import Dataset_service
from nets import NNWF_Net
from datasets import Train_NNWFDataset, Eval_NNWFDataset


print("\nrunning...\n")

# TODO 波高のみの予測にしたので、ServiceのSQL書き換えが必要？？
# TODO transformsのクラスをMainでインスタンス
# TODO Unit test 実装したい
# TODO 気圧と時刻の学習
# TODO early stop 実装したい
# TODO async await 実装するべき


def main():
    # TODO 学習のクラス化
    epochs = 100
    batch_size = 128
    learning_rate = 0.001
    model_name = "nnwf01"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_service = Dataset_service("train")
    eval_service = Dataset_service("eval")

    net = NNWF_Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    loss_hist_model = Loss_hist_model()

    # TODO メゾットとかで、簡素化したい。
    with Train_NNWFDataset(train_service) as train_dataset, \
            Eval_NNWFDataset(eval_service) as eval_dataset:

        print(f"train length:{len(train_dataset)}\n",
              f"eval: length:{len(eval_dataset)}\n")

        real_values = eval_dataset.get_real_values()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

        for epoch in range(1, epochs+1):
            draw_mode = True if epoch % int(
                epochs // 10) == 0 or epoch == 1 else False
            if draw_mode:
                print(f"epoch: {epoch}/{epochs}")

            train_net = net.train()
            count_batches = len(train_dataloader)
            train_loss = 0
            for data, real_val in train_dataloader:
                pred = train_net(data)
                train_loss = loss_func(pred, real_val)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            # TODO compute_extract_loss気に食わん
            extract_height_loss = compute_extract_loss(
                pred, real_val)
            loss_hist_model.waveheight_train.append(extract_height_loss)
            loss_hist_model.train.append(train_loss.item())

            eval_net = net.eval()
            count_batches = len(eval_dataloader)
            height_eval_loss = 0
            eval_loss = 0
            pred_hist = []
            with torch.no_grad():
                for data, real_val in eval_dataloader:
                    pred = eval_net(data)
                    eval_loss += loss_func(pred, real_val).item()
                    extract_height_loss = compute_extract_loss(
                        pred, real_val)
                    height_eval_loss += extract_height_loss

                    if draw_mode:
                        pred_hist.extend(pred.tolist())

            height_eval_loss /= count_batches
            eval_loss /= count_batches
            loss_hist_model.waveheight_eval.append(height_eval_loss)

            loss_hist_model.eval.append(eval_loss)

            if draw_mode:
                draw_predict(real_values, pred_hist, epoch)

                extract_bool = real_values[:, 0] > 0.2
                extract_real_values = real_values[extract_bool]
                extract_pred_hist = torch.Tensor(pred_hist)[extract_bool]
                draw_predict(extract_real_values,
                             extract_pred_hist, "e"+str(epoch))

            # TODO early func 書こう
            if len(loss_hist_model.eval) - loss_hist_model.eval.index(min(loss_hist_model.eval)) > 5:
                print(f"\nOperate early stop epoch: {epoch}\n")

    loss_hist_model.draw_loss(model_name)
    torch.save(net.state_dict(), f"nets/state_dicts/{model_name}.pt")


def compute_extract_loss(pred, real):
    loss_func = nn.MSELoss()
    extract_bool = real[:, 0] > 0.2
    extract_pred = pred[extract_bool]
    extract_real = real[extract_bool]
    height_loss = loss_func(extract_pred[:, 0], extract_real[:, 0]).item()
    return height_loss


def draw_predict(real_values, predHist, epoch):
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.5)

    height_ax = fig.add_subplot(
        111, ylabel="wave height", title=f"epoch: {epoch}")
    height_ax.plot(range(len(real_values)), list(
        map(lambda x: x[0], real_values)), label="observed value")
    height_ax.plot(
        range(len(predHist)), list(map(lambda x: x[0], predHist)),
        label="predicted value", alpha=0.5, color="red")
    height_ax.grid()
    height_ax.legend()

    plt.savefig(f"result/Yt_Yp{epoch}.jpg")
    plt.close(fig)


class Loss_hist_model():
    def __init__(self):
        self.train = []
        self.eval = []
        self.waveheight_train = []
        self.waveheight_eval = []

    def draw_loss(self, model_name):
        fig = plt.figure()
        ax = fig.add_subplot(
            111, title="loss", ylabel="MSE loss", xlabel="epochs")
        self.__plot_loss(ax, self.train, self.eval)
        plt.savefig(f"result/{model_name}_loss.jpg")

        fig = plt.figure()
        height_ax = fig.add_subplot(
            111, ylabel="Height MSE loss", xlabel="epochs")
        self.__plot_loss(height_ax, self.waveheight_train,
                         self.waveheight_eval)
        plt.savefig(f"result/height_loss.jpg")

    def __plot_loss(self, ax, train_loss, eval_loss):
        ax.plot(range(1, len(train_loss)+1), train_loss, label="train")
        ax.plot(range(1, len(eval_loss)+1), eval_loss, label="eval")
        ax.grid()
        ax.legend()


if __name__ == "__main__":
    main()
    print("\nDone!\n")
