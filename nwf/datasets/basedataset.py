import math
import sys

import torch
from torch.utils.data import IterableDataset

from services.recordService import RecordService, RecordServiceModel


class DatasetBaseModel(IterableDataset):
    def __init__(
            self, recordService: RecordService, forecastHour: int, trainHour: int):
        super().__init__()
        self.__recordService = recordService
        self.forecastHour = forecastHour
        self.trainHour = trainHour
        self.recordBuffer = []
        self.len, self.dataSize = self.__shape()

    def __del__(self):
        self.__recordService.close()

    def __shape(self):
        for count, (data, _) in enumerate(self):
            continue
        return count + 1, len(data)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.recordBuffer = []
        self.__recordService.executeSql()
        return self

    def __next__(self):
        self.__loadBuffer()
        data = self.__traindataMolding()
        label = self.__labelMolding()
        return torch.Tensor(data), torch.Tensor(label)

    def __loadBuffer(self):
        """
        DBから1Recordを読み、Bufferにappendする。
        """
        while True:
            # BufferがEmpltyの場合、学習時間分のRecordを読む
            if not self.recordBuffer:
                self.__fillBuffer()
            else:
                self.recordBuffer.pop(0)
                self.recordBuffer.append(self.__recordService.nextRecord())

            if self.__includedInferiority():
                continue

            break

        if len(self.recordBuffer) != self.trainHour + self.forecastHour:
            print("error ", file=sys.stderr)
            sys.exit(1)

    def __fillBuffer(self):
        """
        学習時間分のRecordを読み、recordBufferに格納する。
        """
        if self.recordBuffer:
            print("Error recordBufferが空ではない。", file=sys.stderr)

        for _ in range(self.forecastHour + self.trainHour):
            record = self.__recordService.nextRecord()
            self.recordBuffer.append(record)

    def __includedInferiority(self):
        """
        recordBuffer内に不良Recordが含まれているかを返す
        """
        # traindataのvalidation
        for record in self.recordBuffer[:self.trainHour]:
            for v in record.__dict__.values():
                if v is None:
                    return True

        # labeldataのvalidation
        for v in self.recordBuffer[-1].__dict__.values():
            if v is None:
                return True

        return False

    def __traindataMolding(self) -> list:
        """
        recordBufferに格納されたRecordを学習入力データに整形する。

        Returns:
        -----
            traindata(list): 学習用の整形された入力データ
        """
        data = []
        for index in range(self.trainHour):
            record: RecordServiceModel = self.recordBuffer[index]
            normalize_month = record.datetime.month / 12
            normalize_hour = record.datetime.hour / 24
            sin_month = math.sin(2 * math.pi * normalize_month)
            cos_month = math.cos(2 * math.pi * normalize_month)
            sin_hour = math.sin(2 * math.pi * normalize_hour)
            cos_hour = math.cos(2 * math.pi * normalize_hour)
            isWindWave = record.period > record.height * 4 + 2
            windwave = int(isWindWave)
            swellwave = int(not isWindWave)

            data.extend([
                sin_month,
                cos_month,
                sin_hour,
                cos_hour,

                windwave,
                swellwave,

                record.temperature,

                # record.fukuiPressure,
                # record.fukuyamaPressure,
                # record.hamadaPressure,
                # record.hikonePressure,
                # record.himejiPressure,
                # record.hiroshimaPressure,
                record.kobePressure,
                # record.kochiPressure,
                # record.kurePressure,
                # record.kyotoPressure,
                # record.maizuruPressure,
                # record.matsuePressure,
                # record.matsuyamaPressure,
                # record.murotomisakiPressure,
                # record.naraPressure,
                # record.okayamaPressure,
                # record.osakaPressure,
                # record.owasePressure,
                # record.saigoPressure,
                # record.sakaiPressure,
                # record.shimizuPressure,
                # record.shionomisakiPressure,
                # record.sukumoPressure,
                # record.sumotoPressure,
                # record.tadotsuPressure,
                # record.takamatsuPressure,
                # record.tokushimaPressure,
                # record.tottoriPressure,
                # record.toyookaPressure,
                # record.tsuPressure,
                # record.tsurugaPressure,
                # record.tsuyamaPressure,
                # record.uenoPressure,
                # record.uwajimaPressure,
                # record.wakayamaPressure,
                # record.yokkaichiPressure,
                # record.yonagoPressure,

                record.ukb_velocity,
                record.ukb_sin_direction,
                record.ukb_cos_direction,
                record.kix_velocity,
                record.kix_sin_direction,
                record.kix_cos_direction,
                record.tomogashima_velocity,
                record.tomogashima_sin_direction,
                record.tomogashima_cos_direction,
                record.akashi_velocity,
                record.akashi_sin_direction,
                record.akashi_cos_direction,
                record.osaka_velocity,
                record.osaka_sin_direction,
                record.osaka_cos_direction,

                record.height,
                record.period
            ])

        return data

    def __labelMolding(self) -> list:
        """
        recordBufferに格納されたRecordを学習の真値データ（Label）に整形する。

        Returns:
        -----
            traindata(list): 学習用の整形された真値データ（Label）
        """
        data = []
        record: RecordServiceModel = self.recordBuffer[
            self.trainHour + self.forecastHour - 1]
        data.extend([
            record.height])
        return data

    def close(self):
        self.__recordService.close()

    def inferiorityList(self) -> list:
        self.__iter__()
        inferiorityList = [True for _ in range(self.trainHour)]
        while True:
            try:
                if not self.recordBuffer:
                    self.__fillBuffer()
                else:
                    self.recordBuffer.pop(0)
                    self.recordBuffer.append(self.__recordService.nextRecord())
                inferiorityList.append(self.__includedInferiority())
            except StopIteration:
                break
        return inferiorityList
