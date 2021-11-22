import json
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
section = "learning"


class Config():
    def __init__(self) -> None:
        self.epochs = int(config.get(section, "epochs"))
        self.batchSize = int(config.get(section, "batchsize"))
        self.learningRate = float(config.get(section, "learningrate"))
        self.caseName = config.get(section, "casename")
        self.earlyStopEndure = int(config.get(section, "earlystop_endure"))
        self.targetYear = int(config.get(section, "targetyear"))
        self.trainHour = int(config.get(section, "trainhour"))

    def save(self, savedir):
        with open(savedir+"configLog.json", mode="w") as f:
            json.dump(self.__dict__, f, indent=4)
