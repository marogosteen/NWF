import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from nnwf.learn import LearningModel
from nnwf.net import NNWF_Net
from nnwf.dataset import TrainDatasetModel as Tds, EvalDatasetModel as Eds
from nnwf.history import HistoryModel
from nnwf.report import ReportModel

"""
# TODO 
    Unit test 実装したい
    Datasetのbegin_year,end_yearのエラーハンドリングするべき
"""

config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for year in [2016, 2017, 2018, 2019]:
    config.targetYear = year
    caseName = config.caseName+str(year)
    print(f"\nlearning...{caseName}\n")

    savedir = f"result/{caseName}/"
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with Tds(forecast_hour=config.forecastHour, train_hour=config.trainHour, targetyear=config.targetYear) as train_dataset, \
            Eds(forecast_hour=config.forecastHour, train_hour=config.trainHour, targetyear=config.targetYear) as eval_dataset:
        print(f"train length:{len(train_dataset)}",
              f"eval: length:{len(eval_dataset)}",
              f"input data size:{train_dataset.dataSize}\n", sep="\n")

        transform = transforms.Lambda(lambda x: (
            x - train_dataset.mean)/train_dataset.std)

        trainDataLoader = DataLoader(
            train_dataset, batch_size=config.batchSize)
        evalDataLoader = DataLoader(eval_dataset, batch_size=config.batchSize)

        history = HistoryModel()
        net = NNWF_Net(train_dataset.dataSize).to(device)
        learnigModel = LearningModel(
            net=net,
            optimizer=optim.Adam(net.parameters(), lr=config.learningRate),
            lossFunc=nn.MSELoss(),
            trainDataLoader=trainDataLoader,
            evalDataLoader=evalDataLoader,
            transform=transform,
            earlyStopEndure=config.earlyStopEndure,
        )

        history = learnigModel.fit(config.epochs, history)
        history.showResult()

        nextObserved = np.array(eval_dataset.inferiorityList(), dtype=object)
        nextPredicted = np.copy(nextObserved)

        observed = eval_dataset.observed().numpy()
        nextObserved[nextObserved == False] = observed

        predicted = np.array(learnigModel.bestPredicted(bestModelState=history.bestModelState), dtype=object)
        predicted = predicted.reshape(-1)
        nextPredicted[nextPredicted == False] = predicted

        observed = nextObserved.tolist()
        predicted = nextPredicted.tolist()

        report = ReportModel(config, history, observed, predicted)
        report.save(savedir+caseName+".json")

        # with open(savedir+"/observed.csv", mode="w") as f:
        #     for line in observed:
        #         f.write(",".join(list(map(str, line))) + "\n")

        # with open(savedir+"/predicted.csv", mode="w") as f:
        #     for line in predicted:
        #         f.write(",".join(list(map(str, line))) + "\n")

    torch.save(history.bestModelState, savedir+"/state_dict.pt")
    config.save(savedir)
    history.save_history(caseName)
    history.draw_loss(caseName)


print(f"\nDone! {config.caseName}\n")
