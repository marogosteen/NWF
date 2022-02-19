import datetime

from services.recordServiceModel import RecordServiceModel
from repositories import dbContext


FETCHCHOUNT = 10000


class RecordService():
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

    # def __init__(self, query):
    def __init__(self, targetyear: int, mode: str):
        self.mode = mode
        self.dbContext = dbContext.DbContext()
        self.query = newquery(targetyear, mode)
        self.executeSql()
        self.fetchMany()

    def close(self) -> None:
        """
        databaseのconnectをcloseする．
        """

        self.dbContext.close()

    def __del__(self):
        self.close()

    def executeSql(self) -> None:
        """
        sqlを実行し，self.cursorを初期化する.
        """
        self.dbContext.cursor.execute(self.query)

    def fetchMany(self) -> None:
        """
        databaseからrecordを10000件ずつfetchし，bufferをlist_iterate型でself.__bufferに格納する．

        Raises:
        -----
            StopIteration: fetchの際にrecord数が0件の場合にStopIterrationをCallする．
        """
        buffer = self.dbContext.cursor.fetchmany(FETCHCHOUNT)
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

        return RecordServiceModel(record)

    def leavequery(self, modelname) -> None:
        savepath = f"result/{modelname}/query.txt"
        with open(savepath, mode="w") as f:
            f.write(self.query)


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
    ukb.datetime,

    ukb.velocity,
    ukb.sin_direction,
    ukb.cos_direction,
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
    kobePressure.air_pressure,

    Wave.significant_height,
    Wave.significant_period

FROM
    Wind AS ukb
    INNER JOIN Wind AS kix ON ukb.datetime == kix.datetime
    INNER JOIN Wind AS tomogashima ON ukb.datetime == tomogashima.datetime
    INNER JOIN Wind AS akashi ON ukb.datetime == akashi.datetime
    INNER JOIN Wind AS osaka ON ukb.datetime == osaka.datetime
    INNER JOIN Temperature ON ukb.datetime == Temperature.datetime
    INNER JOIN AirPressure AS kobePressure ON ukb.datetime == kobePressure.datetime
    INNER JOIN Wave ON ukb.datetime == Wave.datetime

WHERE
    ukb.place == 'ukb' AND
    kix.place == 'kix' AND
    tomogashima.place == 'tomogashima' AND
    akashi.place == 'akashi' AND
    osaka.place == 'osaka' AND
    kobePressure.place == 'kobe' AND
    {yes_or_no}(
        datetime(ukb.datetime) >= datetime("{targetyear}-01-01 00:00") AND
        datetime(ukb.datetime) <= datetime("{targetyear}-12-31 23:00")
    )

ORDER BY
    ukb.datetime
;
"""
