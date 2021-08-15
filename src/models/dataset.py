import re
import datetime


def interpretData(nowphasLine:str):
    timeSeries = datetime.datetime(
        year=int(nowphasLine[:4]), month=int(nowphasLine[4:6]), day=int(nowphasLine[6:8]),
        hour=int(nowphasLine[8:10]), minute=int(nowphasLine[10:12])
    )
    perLineData = re.split(" +", nowphasLine[11:])
