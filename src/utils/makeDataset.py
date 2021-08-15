import io
import re
import math
import datetime
"""
このファイルは，アメダスとナウファスのデータから学習データの生成を行います．
保存される学習データの形式はCSVです．学習データの内容は以下の通りです．
    
    緯度方向の風速,経度方向の風速,気温,気圧,有義波高

以下のPathを確認してから実行してください．
データセットディレクトリ: datasetDir
書き込み先ディレクトリ: writeDir
アメダスファイル名: amedasFileName
ナウファスファイル名: nowphasFileName
"""
def interpretNowphasLine(nowphasLine:str) -> list:
    """
    テキスト形式であるnowphasデータの各行を読み込む．

    Args:
        nowphasLine (str): 
            ファイルの一行
    
    Returns:
        list:
            nowphasデータの一行．
    """    
    timeSeries = nowphasLine[:12]
    perLineData:list = re.split(" +", nowphasLine[11:])

    return [timeSeries]+perLineData

def convertRadian(direction) -> float:
    """
    strの16方位からRadianを返す．

    Args:
        direction (str): 
            strの方位

    Returns:
        float: 
            16方位からRadianに直したfloatの値
    """

    indexDirection = [
        "東","東南東","南東","南南東",
        "南","南南西","南西","西南西",
        "西","西北西","北西","北北西",
        "北","北北東","北東","東北東",
    ].index(direction)
    radian = (16 - indexDirection) / 16 * 2 * math.pi

    return radian

def skipLines(file:io.TextIOWrapper,times):
    """
    ファイル読み込み時に読み込みをスキップする．

    Args:
        file (io.TextIOWrapper): [description]
            読み込みファイル
        times ([type]): [description]
            スキップする行数
    """    
    for i in range(times):
        file.readline()

writeFilePath:str = "dataset/2019_01.csv"
amedasFilePath:str = "src/utils/data/amedas2019.csv"
nowphasFilePath:str = "src/utils/data/h306e019.txt"

with open(writeFilePath, encoding="utf-8", mode="w") as wf:
    with open(amedasFilePath, encoding="shift_jis", mode="r") as amedasFile:
        with open(nowphasFilePath, encoding="shift_jis", mode="r") as nowphasFile:
            skipLines(amedasFile,6)
            skipLines(nowphasFile,1)
            while True:
                amedasLine:list = amedasFile.readline().split(",")
                try:
                    amedasTimeSeries:datetime.datetime = datetime.datetime.strptime(
                        amedasLine[0], "%Y/%m/%d %H:%M:%S"
                    )
                except ValueError:
                    print("datetimeのformatが一致しない datetime:",amedasLine[0])

                while True:
                    nowphasLine:list = interpretNowphasLine(nowphasFile.readline())
                    nowphasTimeSeries:datetime.datetime = datetime.datetime(
                        year=int(nowphasLine[0][:4]), month=int(nowphasLine[0][4:6]), 
                        day=int(nowphasLine[0][6:8]), hour=int(nowphasLine[0][8:10]), 
                        minute=int(nowphasLine[0][10:12])
                    )
                    if amedasTimeSeries == nowphasTimeSeries:
                        lineTimeSeries:datetime.datetime = amedasTimeSeries
                        break

                if (amedasLine[2] == amedasLine[5] == amedasLine[7] == amedasLine[10] == "8" ) == False:
                    print("アメダスのデータが欠損している:",lineTimeSeries)
                    continue
                direction = amedasLine[6]
                if direction == "静穏":
                    print("風速が小さい:",direction)
                    continue
                if nowphasLine[5] == "999":
                    print("ナウファスのデータが欠損している:",lineTimeSeries)
                    continue

                velocity:float = float(amedasLine[4])
                temperature:float = float(amedasLine[1])
                airPressure:float = float(amedasLine[9])
                radian:float = convertRadian(direction)

                longitudeVelocity:float = round(velocity * math.sin(radian), 12)
                latitudeVelocity:float = round(velocity * math.cos(radian), 12)
                significantWaveHeight:float = nowphasLine[5]

                wf.write(
                    "{},{},{},{},{}\n".format(
                        longitudeVelocity, 
                        latitudeVelocity, 
                        temperature, 
                        airPressure, 
                        significantWaveHeight
                    )
                )

                if lineTimeSeries == datetime.datetime(year=2019, month=12, day=31, hour=23):
                    break

print("\ncompleted\n")