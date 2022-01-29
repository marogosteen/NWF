import datetime
import sqlite3


class DbFetcher():
    """
    Databaseから観測値をfetchするClass

    Args:
    -----
        - targetyear (int): 予測対象年
        - mode (str): modeは"train"または"eval"．
        "tarain"の場合はtargetyear以外の期間からrecordをfetchする．
        "eval"の場合はtargetyearの年からrecordをfetchする．

    Raises:
    -----
        StopIteration: fetchMany()で返されたlistの要素数が0件の場合はStopIterrationをCallする．
    """

    dbPath = "database/dataset.sqlite"

    def __init__(self, targetyear: int, mode: str):
        self.mode = mode
        self.targetyear = targetyear
        self.connect = sqlite3.connect("database/dataset.sqlite")
        self.cursor = self.connect.cursor()
        self.query = newquery(targetyear, mode)
        self.executeSql()

    def __del__(self):
        self.close()

    def close(self) -> None:
        """
        databaseのconnectをcloseする．
        """

        self.connect.close()

    def executeSql(self) -> None:
        """
        sqlを実行し，self.cursorを初期化する.
        """

        self.cursor = self.cursor.execute(self.query)
        self.fetchMany()

    def fetchMany(self) -> None:
        """
        databaseからrecordを10000件ずつfetchし，bufferをlist_iterate型でself.__bufferに格納する．

        Raises:
        -----
            StopIteration: fetchの際にrecord数が0件の場合にStopIterrationをCallする．
        """

        buffer = self.cursor.fetchmany(10000)
        if not buffer:
            raise StopIteration
        self.__buffer = iter(buffer)

    def nextRecord(self):
        """
        list_iterate型のbufferからnextによって1record返す．\n
        bufferの要素がemptyの場合にnextすると，次のbufferをfetchする．

        Returns:
        -----
            record (RecordModel): 1record
        """

        try:
            record = next(self.__buffer)
        except StopIteration:
            self.fetchMany()
            record = next(self.__buffer)

        return RecordModel(record)

    def leavequery(self, modelname) -> None:
        savepath = f"result/{modelname}/query.txt" 
        with open(savepath, mode="w") as f:
            f.write(self.query)



class RecordModel():
    """
    1時刻あたりの観測データ
    """

    def __init__(self, record):
        self.datetime = datetime.datetime.strptime(record[0], "%Y-%m-%d %H:%M")
        self.kobe_velocity = record[1]
        self.kobe_sin_direction = record[2]
        self.kobe_cos_direction = record[3]
        self.kix_velocity = record[4]
        self.kix_sin_direction = record[5]
        self.kix_cos_direction = record[6]
        self.tomogashima_velocity = record[7]
        self.tomogashima_sin_direction = record[8]
        self.tomogashima_cos_direction = record[9]
        self.akashi_velocity = record[10]
        self.akashi_sin_direction = record[11]
        self.akashi_cos_direction = record[12]
        self.osaka_velocity = record[13]
        self.osaka_sin_direction = record[14]
        self.osaka_cos_direction = record[15]
        self.temperature = record[16]
        self.air_pressure = record[17]
        self.height = record[18]
        self.period = record[19]


def newquery(targetyear: int, mode: str) -> str:
    """
    fetchする期間を指定し，sql queryを返す．

    Returns:
    -----
        - targetyear (int): 予測対象年
        - mode (str): modeは"train"または"eval"．
        "tarain"の場合はtargetyear以外の期間からrecordをfetchする．
        "eval"の場合はtargetyearの年からrecordをfetchする．
    """

    if mode == "train":
        yes_or_no = "NOT"
    elif mode == "eval":
        yes_or_no = ""
    else:
        errmse = f'modeとして {mode} は受け付けません．modeは"train"または"eval"で指定してください．'
        exit(errmse)

    return f"""
SELECT
    kobe.datetime,

    kobe.velocity,
    kobe.sin_direction,
    kobe.cos_direction,
    kix.velocity,
    kix.sin_direction,
    kix.cos_direction,
    tomogashima.velocity,
    tomogashima.sin_direction,
    tomogashima.cos_direction,
    akashi.velocity,
    akashi.sin_direction,
    akashi.cos_direction,
    osaka.velocity,
    osaka.sin_direction,
    osaka.cos_direction,

    Temperature.temperature,
    AirPressure.air_pressure,

    Wave.significant_height,
    Wave.significant_period

FROM
    Wind AS kobe
    INNER JOIN Wind AS kix ON kobe.datetime == kix.datetime
    INNER JOIN Wind AS tomogashima ON kobe.datetime == tomogashima.datetime
    INNER JOIN Wind AS akashi ON kobe.datetime == akashi.datetime
    INNER JOIN Wind AS osaka ON kobe.datetime == osaka.datetime
    INNER JOIN Temperature ON kobe.datetime == Temperature.datetime
    INNER JOIN AirPressure ON kobe.datetime == AirPressure.datetime
    INNER JOIN Wave ON kobe.datetime == Wave.datetime

WHERE
    kobe.place == 'kobe' AND
    kix.place == 'kix' AND
    tomogashima.place == 'tomogashima' AND
    akashi.place == 'akashi' AND
    osaka.place == 'osaka' AND
    {yes_or_no}(
        datetime(kobe.datetime) >= datetime("{targetyear}-01-01 00:00") AND
        datetime(kobe.datetime) <= datetime("{targetyear}-12-31 23:00")
    )

ORDER BY
    kobe.datetime
;
"""
