import datetime

from sqlalchemy import create_engine
from sqlalchemy import orm, select, Column, INTEGER, REAL, TEXT
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import or_


Base = declarative_base()
Datatime = DATETIME(
    storage_format="%(year)04d-%(month)02d-%(day)02d %(hour)02d:%(minute)02d",
    regexp=r"(\d+)-(\d+)-(\d+) (\d+):(\d+)")
dbPath = "sqlite:///database/dataset.sqlite"


class WindTb(Base):
    __tablename__ = "Wind"
    datetime = Column(Datatime, primary_key=True)
    place = Column(TEXT, primary_key=True)
    inferiority = Column(INTEGER, primary_key=True)
    velocity = Column(REAL, primary_key=True)
    sin_direction = Column(REAL, primary_key=True)
    cos_direction = Column(REAL, primary_key=True)


class WaveTb(Base):
    __tablename__ = "Wave"
    datetime = Column(Datatime, primary_key=True)
    place = Column(TEXT, primary_key=True)
    inferiority = Column(INTEGER, primary_key=True)
    mean_height = Column(REAL, primary_key=True)
    mean_period = Column(REAL, primary_key=True)
    significant_height = Column(REAL, primary_key=True)
    significant_period = Column(REAL, primary_key=True)
    ten_percent_height = Column(REAL, primary_key=True)
    ten_percent_period = Column(REAL, primary_key=True)
    max_height = Column(REAL, primary_key=True)
    max_period = Column(REAL, primary_key=True)
    direction = Column(INTEGER, primary_key=True)


class AirPressureTb(Base):
    __tablename__ = "AirPressure"
    datetime = Column(Datatime, primary_key=True)
    place = Column(TEXT, primary_key=True)
    inferiority = Column(INTEGER, primary_key=True)
    air_pressure = Column(REAL, primary_key=True)


class TemperatureTb(Base):
    __tablename__ = "Temperature"
    datetime = Column(Datatime, primary_key=True)
    place = Column(TEXT, primary_key=True)
    inferiority = Column(INTEGER, primary_key=True)
    temperature = Column(REAL, primary_key=True)


kobe: WindTb = orm.aliased(WindTb, name="kobe")
kix: WindTb = orm.aliased(WindTb, name="kix")
tomogashima: WindTb = orm.aliased(WindTb, name="tomogashima")
akashi: WindTb = orm.aliased(WindTb, name="akashi")
osaka: WindTb = orm.aliased(WindTb, name="osaka")


class DbService():
    def __init__(self, query):
        engine = create_engine(dbPath, echo=False)
        SessionClass = orm.sessionmaker(engine)
        self.__session: orm.session.Session = SessionClass()
        self.__query = query
        self.__sqlResult = self.__session.execute(query)

    def close(self):
        self.__session.close()

    def initQuery(self):
        self.__sqlResult = self.__session.execute(self.__query)
        self.fetchMany()

    def fetchMany(self):
        buffer = self.__sqlResult.fetchmany(10000)
        if not buffer:
            raise StopIteration
        self.__buffer = iter(buffer)

    def nextRecord(self) -> list:
        try:
            record = next(self.__buffer)
        except StopIteration:
            self.fetchMany()
            record = next(self.__buffer)

        return record


def get_train_sqlresult(targetyear: int):
    return basequery().where(
        or_(
            kobe.datetime < datetime.date(targetyear, 1, 1),
            kobe.datetime > (datetime.date(targetyear, 12, 31) + datetime.timedelta(days=1)))
    ).order_by(kobe.datetime)


def getEvalSqlresult(targetyear: int):
    return basequery().where(
        kobe.datetime >= datetime.date(targetyear, 1, 1),
        kobe.datetime < (datetime.datetime(targetyear, 12, 31) + datetime.timedelta(days=1))
    ).order_by(kobe.datetime)


def basequery():
    return select(
        (kobe.inferiority).label("kobe_inferiority"),
        (kix.inferiority).label("kix_inferiority"),
        (tomogashima.inferiority).label("tomogashima_inferiority"),
        (akashi.inferiority).label("akashi_inferiority"),
        (osaka.inferiority).label("osaka_inferiority"),
        (TemperatureTb.inferiority).label("temperature_inferiority"),
        (AirPressureTb.inferiority).label("airPressure_inferiority"),
        (WaveTb.inferiority).label("wave_inferiority"),

        (kobe.datetime).label("datetime"),

        (kobe.velocity).label("kobe_velocity"),
        (kobe.sin_direction).label("kobe_sinDirection"),
        (kobe.cos_direction).label("kobe_cosDirection"),
        (kix.velocity).label("kix_velocity"),
        (kix.sin_direction).label("kix_sinDirection"),
        (kix.cos_direction).label("kix_cosDirection"),
        (tomogashima.velocity).label("tomogashima_velocity"),
        (tomogashima.sin_direction).label("tomogashima_sinDirection"),
        (tomogashima.cos_direction).label("tomogashima_cosDirection"),
        (akashi.velocity).label("akashi_velocity"),
        (akashi.sin_direction).label("akashi_sinDirection"),
        (akashi.cos_direction).label("akashi_cosDirection"),
        (osaka.velocity).label("osaka_velocity"),
        (osaka.sin_direction).label("osaka_sinDirection"),
        (osaka.cos_direction).label("osaka_cosDirection"),

        (TemperatureTb.temperature).label("temperature"),
        (AirPressureTb.air_pressure).label("air_pressure"),
        (WaveTb.significant_height).label("height"),
        (WaveTb.significant_period).label("period")
    )\
        .join(kix, kobe.datetime == kix.datetime)\
        .join(tomogashima, kobe.datetime == tomogashima.datetime)\
        .join(akashi, kobe.datetime == akashi.datetime)\
        .join(osaka, kobe.datetime == osaka.datetime)\
        .join(AirPressureTb, kobe.datetime == AirPressureTb.datetime)\
        .join(TemperatureTb, kobe.datetime == TemperatureTb.datetime)\
        .join(WaveTb, kobe.datetime == WaveTb.datetime)\
        .where(
            kobe.place == "kobe",
            kix.place == "kix",
            tomogashima.place == "tomogashima",
            akashi.place == "akashi",
            osaka.place == "osaka",
            TemperatureTb.place == "kobe")
