from services.recordServiceModel import RecordServiceModel
from services.query import DbQuery
from repositories import dbContext


FETCHCHOUNT = 10000


class RecordService():
    """
    Databaseから観測値をfetchするClass

    Args:
    -----
        - query (DbQuery)

    Raises:
    -----
        StopIteration: fetchMany()で返されたlistの要素数が0件の場合はStopIterrationをCallする．
    """

    def __init__(self, query: DbQuery) -> None:
        self.query = query.query
        self.dbContext = dbContext.DbContext()
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
