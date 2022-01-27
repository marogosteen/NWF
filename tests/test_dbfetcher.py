import datetime
import unittest

from nnwf.dbfetcher import DbFetcher
from nnwf.dbfetcher import RecordModel


class TestDbFetcher(unittest.TestCase):
    def test_trainfetch(self):
        for targetyear in range(2016, 2020):
            comfirmtime = datetime.datetime(2016, 1, 1, 0, 0)
            duration = datetime.timedelta(hours=1)

            fetcher = DbFetcher(targetyear, "train")
            record: RecordModel
            while True:
                if comfirmtime.year == targetyear:
                    comfirmtime = datetime.datetime(targetyear + 1, 1, 1, 0, 0)
                try:
                    record = fetcher.nextRecord()
                except StopIteration:
                    break

                errmse = f"recordtimeがcomfirmtimeと一致しない． comfirmtime: {comfirmtime}, record.datetime: {record.datetime}"
                self.assertTrue(comfirmtime == record.datetime, errmse)
                comfirmtime += duration

    def test_evalfetch(self):
        for targetyear in range(2016, 2020):
            comfirmtime = datetime.datetime(targetyear, 1, 1, 0, 0)
            duration = datetime.timedelta(hours=1)

            fetcher = DbFetcher(targetyear, "eval")
            record: RecordModel
            while True:
                try:
                    record = fetcher.nextRecord()
                except StopIteration:
                    break

                errmse = f"recordtimeがcomfirmtimeと一致しない． comfirmtime: {comfirmtime}, record.datetime: {record.datetime}"
                self.assertTrue(comfirmtime == record.datetime, errmse)
                comfirmtime += duration
