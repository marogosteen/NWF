from datetime import datetime


class RecordServiceModel():
    """
    1時刻あたりの観測データ
    """

    def __init__(self, record):
        self.datetime = datetime.strptime(record[0], "%Y-%m-%d %H:%M")
        self.ukb_velocity = record[1]
        self.ukb_sin_direction = record[2]
        self.ukb_cos_direction = record[3]
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
