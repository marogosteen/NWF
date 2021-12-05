import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from nnwf.learn import LearningModel
from nnwf.net import NNWF_Net
from nnwf.dataset import TrainDatasetModel, EvalDatasetModel
from nnwf.log import LogModel


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

    with TrainDatasetModel(forecast_hour=config.forecastHour, train_hour=config.trainHour, targetyear=config.targetYear) as train_dataset, \
            EvalDatasetModel(forecast_hour=config.forecastHour, train_hour=config.trainHour, targetyear=config.targetYear) as eval_dataset:
        print(f"train length:{len(train_dataset)}",
              f"eval: length:{len(eval_dataset)}",
              f"input data size:{train_dataset.dataSize}\n", sep="\n")

        transform = transforms.Lambda(lambda x: (
            x - train_dataset.mean)/train_dataset.std)
        real_values = eval_dataset.getRealValues()

        trainDataLoader = DataLoader(
            train_dataset, batch_size=config.batchSize)
        evalDataLoader = DataLoader(eval_dataset, batch_size=config.batchSize)

        logModel = LogModel()
        net = NNWF_Net(train_dataset.dataSize, config.forecastHour).to(device)
        learnigModel = LearningModel(
            net=net,
            optimizer=optim.Adam(net.parameters(), lr=config.learningRate),
            lossFunc=nn.MSELoss(),
            trainDataLoader=trainDataLoader,
            evalDataLoader=evalDataLoader,
            transform=transform,
            earlyStopEndure=config.earlyStopEndure,
        )

        logModel = learnigModel.fit(config.epochs, logModel)
        logModel.showResult()
        pred = learnigModel.getBestPredictedValue(logModel.bestModelState)

        with open(savedir+"/observed.csv", mode="w") as f:
            for line in real_values.tolist():
                f.write(",".join(list(map(str, line))) + "\n")

        with open(savedir+"/predicted.csv", mode="w") as f:
            for line in pred:
                f.write(",".join(list(map(str, line))) + "\n")

    torch.save(logModel.bestModelState, savedir+"/state_dict.pt")
    config.save(savedir)
    logModel.save_log(caseName)
    logModel.draw_loss(caseName)


print(f"\nDone! {config.caseName}\n")
