import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from nnwf.nets import NNWF_Net
from nnwf.datasets import Train_NNWFDataset, Eval_NNWFDataset
from nnwf.learning_log import LearningLog


def early_stop_detect(log_model: LearningLog, endure: int):
    current_epoch = len(log_model.train_loss_hist)
    return endure < (current_epoch - log_model.best_epoch())


# TODO Unit test 実装したい
# TODO 気圧の学習
# TODO async await 実装するべき??
# TODO Datasetのbegin_year,end_yearのエラーハンドリングするべき


def main():
    print("\nlearning...\n")

    # TODO 学習のクラス化
    epochs = 150
    batch_size = 128
    learning_rate = 0.001
    model_name = "nnwf"
    early_stop_endure = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = NNWF_Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    log_model = LearningLog()

    # TODO メゾットとかで、簡素化したい。
    with Train_NNWFDataset(forecast_hour=1, train_hour=2, begin_year=2016, end_year=2018) as train_dataset, \
            Eval_NNWFDataset(forecast_hour=1, train_hour=2, begin_year=2019, end_year=2019) as eval_dataset:
        print(f"train length:{len(train_dataset)}\n",
              f"eval: length:{len(eval_dataset)}\n")

        transform = transforms.Lambda(lambda x: (
            x - train_dataset.mean)/train_dataset.std)
        real_values = eval_dataset.get_real_values()

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

        for _ in tqdm.tqdm((range(1, epochs+1))):
            # train lossをbatch毎に求めて最適化する。
            train_net = net.train()
            count_batches = len(train_dataloader)
            train_loss = 0
            for data, real_val in train_dataloader:
                data = transform(data)
                batch_pred = train_net(data)
                train_loss = loss_func(batch_pred, real_val)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            log_model.train_loss_hist.append(train_loss.item())

            # eval lossをbatch毎に求め、総和平均をeval lossとして扱う
            eval_net = net.eval()
            count_batches = len(eval_dataloader)
            eval_loss = 0
            with torch.no_grad():
                for data, real_val in eval_dataloader:
                    data = transform(data)
                    batch_pred = eval_net(data)
                    eval_loss += loss_func(batch_pred, real_val).item()

            eval_loss /= count_batches
            log_model.eval_loss_list.append(eval_loss)

            if log_model.best_loss() > eval_loss or not log_model.best_model_state:
                log_model.best_model_state = net.state_dict()

            if early_stop_detect(log_model, early_stop_endure):
                print("[ Early Stop ]\n")

        print("",
              "best epoch is ", f"\t{log_model.best_epoch()}",
              "best loss is ", f"\t{round(log_model.best_loss(), 5)}",
              sep="\n")

        net.load_state_dict(log_model.best_model_state)
        net.eval()
        eval_loss = 0
        best_pred_list = []
        with torch.no_grad():
            for data, real_val in eval_dataloader:
                data = transform(data)
                batch_pred = net(data)
                best_pred_list.extend(batch_pred.tolist())

        with open("result/observed.csv", mode="w") as f:
            for line in real_values.tolist():
                f.write(str(line[0]) + "\n")

        with open("result/predicted.csv", mode="w") as f:
            for line in best_pred_list:
                f.write(str(line[0]) + "\n")

    model_state_path = f"nnwf/nets/state_dicts/{model_name}.pt"
    log_model.save_best_model_state(model_state_path)
    log_model.save_log(model_name)
    log_model.draw_loss(model_name)


if __name__ == "__main__":
    main()
    print("\nDone!\n")
