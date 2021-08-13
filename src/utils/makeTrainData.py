import math
"""
このファイルは，学習データの生成のためにアメダスのデータから変換と保存を行います．
保存される学習データの形式はCSVです．学習データのHeaderは次の通りです．
    
    日付,緯度方向の風速,経度方向の風速,気温,気圧

実行前にデータの読み込み先と書き込み先を指定してください．
読み込み先変数: writeFilePath
書き込み先変数: readFilePath
"""

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
        "北","北北東","北東","東北東",
        "東","東南東","南東","南南東",
        "南","南南西","南西","西南西",
        "西","西北西","北西","北北西"
    ].index(direction)
    radian = (indexDirection + 12) / 16 * 2 * math.pi

    return radian


writeFilePath:str = "data/kobe2019/traindata1.csv"
readFilePath:str = "data/kobe2019/amedas.csv"

with open(writeFilePath, encoding="utf-8", mode="w") as wf:
    wf.write("date,longitudeVelocity,latitudeVelocity,temperature,airPressure\n")
    with open(readFilePath, encoding="Shift-jis", mode="r") as rf:
        for line in rf.readlines()[6:]:
            line = line.split(",")
            timeSeries:str = line[0]
            if (line[2] == line[5] == line[7] == line[10] == "8" ) == False:
                print("データが欠損している:",timeSeries)
                continue

            direction = line[6]
            if direction == "静穏":
                print("風速が小さい:",direction)
                continue

            velocity:float = float(line[4])
            temperature:float = float(line[1])
            airPressure:float = float(line[9])
            radian:float = convertRadian(direction)

            longitudeVelocity:float =velocity * math.sin(radian) 
            latitudeVelocity:float = velocity * math.cos(radian)

            wf.write(f"{timeSeries},{longitudeVelocity},{latitudeVelocity},{temperature},{airPressure}\n")