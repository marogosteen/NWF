import json
import math

from config import Config
from nwf.history import HistoryModel


class ReportModel():
    def __init__(
            self, casename, cfg: Config, history: HistoryModel,
            observed: list, predicted: list):

        self.caseName: str = casename
        self.epochs: int = cfg.epochs
        self.batchSize: int = cfg.batchSize
        self.learningRate: float = cfg.learningRate
        self.earlyStopEndure: int = cfg.earlyStopEndure
        self.targetYear: int = cfg.targetYear
        self.trainHour: int = cfg.trainHour
        self.forecastHour: int = cfg.forecastHour

        self.inferiorityCount: int = observed.count(None)
        self.dataCount: int = len(observed) - self.inferiorityCount
        self.bestEpoch: int = history.best_epoch()
        self.bestStd: float = round(math.sqrt(history.best_loss()), 3)

        self.observed: list = observed
        self.predicted: list = predicted

    def save(self, path: str):
        savedict = self.__dict__.copy()
        with open(path, 'w') as f:
            f.write(json.dumps(savedict, indent=4))
