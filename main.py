import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from nwf.learn import LearningModel
from nwf.net import NNWF_Net
from nwf.datasets.dataset import TrainDatasetModel, EvalDatasetModel
from nwf.datasets.dbfetcher import DbFetcher
from nwf.history import HistoryModel
from nwf.report import ReportModel

"""
# TODO 
    Datasetのbegin_year,end_yearのエラーハンドリングするべき
    deploy作ろ
    Test
        DataSetの動作確認用のTest書きたい
        bufferの動作確認したい
        Bufの使っていない中間のRecord無視できているかTest
        DBの中身のTestしたい。だぶりとか。
    試すやつ
        db書き換え
        前ReportのRun
        periodのRun
        XAI
        気圧入れる
        transformの有無
    trainを使ったNewアリストのチェック。グラフで。
"""

config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for forecastHour in range(6, 11):
    for year in [2016, 2017, 2018, 2019]:
        config.targetYear = year
        config.forecastHour = forecastHour
        caseName = config.caseName + \
            str(config.trainHour)+"HTrain"+str(forecastHour)+"HLater"+str(year)
        print(f"\nlearning...{caseName}\n")

        savedir = f"result/{caseName}/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        trainfetcher = DbFetcher(targetyear=config.targetYear, mode="train")
        evalfetcher = DbFetcher(targetyear=config.targetYear, mode="eval")
        tds = TrainDatasetModel(
            forecastHour=forecastHour,
            trainHour=config.trainHour,
            fetcher=trainfetcher)
        eds = EvalDatasetModel(
            forecastHour=forecastHour,
            trainHour=config.trainHour,
            fetcher=evalfetcher)

        with tds as trainDataset, eds as evalDataset:
            print(f"train length:{len(trainDataset)}",
                  f"eval: length:{len(evalDataset)}",
                  f"input data size:{trainDataset.dataSize}\n", sep="\n")

            transform = transforms.Lambda(
                lambda x: (x - trainDataset.mean)/trainDataset.std)
            trainDataLoader = DataLoader(
                trainDataset, batch_size=config.batchSize)
            evalDataLoader = DataLoader(
                evalDataset, batch_size=config.batchSize)
            history = HistoryModel()
            net = NNWF_Net(trainDataset.dataSize).to(device)

            learnigModel = LearningModel(
                net=net,
                optimizer=optim.Adam(net.parameters(), lr=config.learningRate),
                lossFunc=nn.MSELoss(),
                trainDataLoader=trainDataLoader,
                evalDataLoader=evalDataLoader,
                transform=transform,
                earlyStopEndure=config.earlyStopEndure)
            history = learnigModel.fit(config.epochs, history)
            history.showResult()

            nextObserved = np.array(
                evalDataset.inferiorityList(), dtype=object)
            nextPredicted = np.copy(nextObserved)

            observed = evalDataset.observed().numpy()
            nextObserved[nextObserved == True] = None
            nextObserved[nextObserved == False] = observed

            predicted = np.array(learnigModel.bestPredicted(
                bestModelState=history.bestModelState), dtype=object)
            predicted = predicted.reshape(-1)
            nextPredicted[nextPredicted == True] = None
            nextPredicted[nextPredicted == False] = predicted

            observed = nextObserved.tolist()
            predicted = nextPredicted.tolist()

            report = ReportModel(
                caseName, config, history, observed, predicted)
            report.save(savedir+caseName+".json")

        torch.save(history.bestModelState, savedir+"/state_dict.pt")
        config.save(savedir)
        history.draw_loss(caseName)
        trainfetcher.leavequery(caseName)


print(f"\nDone! {config.caseName}\n")
