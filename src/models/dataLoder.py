import re
import datetime

class DataLoder():
    # TODO 未完
    def readAmedas(readFilePath:str) -> list:
        data:list = []
        with open(readFilePath, encoding="Shift-jis", mode="r") as f:
            for line in f.readlines()[6:]:
                print(line)
        return data
    # TODO 未完
    def readNowphas(readFilePath:str) -> list:
        data:list = []
        with open(readFilePath, encoding="Shift-jis", mode="r") as f:
            for line in f.readlines()[1:]:
                strTimeSeries = line[:12]
                perLineData = line[11:]
                perLineData = re.split(" +",perLineData)

                timeSeries = datetime.datetime(
                    year=int(strTimeSeries[:4]), month=int(strTimeSeries[4:6]), day=int(strTimeSeries[6:8]),
                    hour=int(strTimeSeries[8:10]), minute=int(strTimeSeries[10:12])
                )

                if timeSeries.minute == 0:
                    pass
        return data