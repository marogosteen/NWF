import copy

import tqdm
import torch

from nnwf.log import LogModel


class LearningModel():
    def __init__(
        self, net, lossFunc, optimizer,
        trainDataLoader, evalDataLoader, transform,
        earlyStopEndure
    ) -> None:
        self.net = net
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.trainDataLoader = trainDataLoader
        self.evalDataLoader = evalDataLoader
        self.transform = transform
        self.earlyStopEndure = earlyStopEndure

    def fit(self, epochs:int, logModel: LogModel) -> LogModel:
        for epoch in tqdm.tqdm(range(1, epochs+1)):
            trainLoss = self.__train(self.trainDataLoader, self.transform)
            evalLoss = self.__eval(self.evalDataLoader, self.transform)
            logModel.train_loss_hist.append(trainLoss)
            logModel.eval_loss_list.append(evalLoss)

            if logModel.isBestLoss(evalLoss):
                logModel.bestEpoch = epoch
                logModel.bestLoss = evalLoss
                logModel.bestModelState = copy.deepcopy(self.net.state_dict())

            if self.__earlyStopDetect(logModel, self.earlyStopEndure):
                # TODO EarlyStop実装
                print("[ Early Stop ]\n")

        return logModel

    def getBestPredictedValue(self, bestModelState) -> list:
        self.net.load_state_dict(bestModelState)
        self.net.eval()
        bestPredList = []
        with torch.no_grad():
            for data, _ in self.evalDataLoader:
                data = self.transform(data)
                batch_pred = self.net(data)
                bestPredList.extend(batch_pred.tolist())
        return bestPredList

    def __train(self, dataloader, transform):
        # train lossをbatch毎に求めて最適化する。
        train_net = self.net.train()
        loss = 0
        for data, real_val in dataloader:
            data = transform(data)
            batch_pred = train_net(data)
            loss = self.lossFunc(batch_pred, real_val)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def __eval(self, dataloader, transform):
        # eval lossをbatch毎に求め、平均をeval lossとして扱う
        eval_net = self.net.eval()
        count_batches = len(dataloader)
        loss = 0
        with torch.no_grad():
            for data, real_val in dataloader:
                data = transform(data)
                batch_pred = eval_net(data)
                loss += self.lossFunc(batch_pred, real_val).item()
        loss /= count_batches

        return loss

    def __earlyStopDetect(self, log_model: LogModel, endure: int):
        current_epoch = len(log_model.train_loss_hist)
        return endure < (current_epoch - log_model.best_epoch())
