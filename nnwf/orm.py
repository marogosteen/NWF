import datetime

from sqlalchemy import orm, select, Column, INTEGER, REAL, TEXT
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import or_


Base = declarative_base()
Datatime = DATETIME(
    storage_format="%(year)04d-%(month)02d-%(day)02d %(hour)02d:%(minute)02d",
    regexp=r"(\d+)-(\d+)-(\d+) (\d+):(\d+)")


class WindTb(Base):
    __tablename__ = "Wind"
    datetime = Column(Datatime, primary_key=True)
    place = Column(TEXT, primary_key=True)
    inferiority = Column(INTEGER, primary_key=True)
    latitude_velocity = Column(REAL, primary_key=True)
    longitude_velocity = Column(REAL, primary_key=True)


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


kobe = orm.aliased(WindTb, name="kobe")
kix = orm.aliased(WindTb, name="kix")
tomogashima = orm.aliased(WindTb, name="tomogashima")
akashi = orm.aliased(WindTb, name="akashi")
awaji = orm.aliased(WindTb, name="awaji")
nishinomiya = orm.aliased(WindTb, name="nishinomiya")
osaka = orm.aliased(WindTb, name="osaka")


def get_train_sqlresult(targetyear: int):
    query = basequery().filter(
        kobe.place == "kobe",
        kix.place == "kix",
        tomogashima.place == "tomogashima",
        akashi.place == "akashi",
        awaji.place == "awaji",
        nishinomiya.place == "nishinomiya",
        osaka.place == "osaka",
        or_(
            kobe.datetime < datetime.datetime(targetyear, 1, 1),
            kobe.datetime > datetime.datetime(targetyear, 12, 31)))\
        .order_by(kobe.datetime)

    return query


def get_eval_sqlresult(targetyear: int):
    query = basequery().filter(
        kobe.place == "kobe",
        kix.place == "kix",
        tomogashima.place == "tomogashima",
        akashi.place == "akashi",
        awaji.place == "awaji",
        nishinomiya.place == "nishinomiya",
        osaka.place == "osaka",
        kobe.datetime >= datetime.datetime(targetyear, 1, 1),
        kobe.datetime <= datetime.datetime(targetyear, 12, 31))\
        .order_by(kobe.datetime)

    return query


def basequery():
    return select(
        (kobe.inferiority).label("kobe_inferiority"),
        (kix.inferiority).label("kix_inferiority"),
        (tomogashima.inferiority).label("tomogashima_inferiority"),
        (akashi.inferiority).label("akashi_inferiority"),
        (awaji.inferiority).label("awaji_inferiority"),
        (nishinomiya.inferiority).label("nishinomiya_inferiority"),
        (osaka.inferiority).label("osaka_inferiority"),
        (WaveTb.inferiority).label("wave_inferiority"),
        (kobe.datetime).label("datetime"),
        (kobe.latitude_velocity).label("kobe_latitude_velocity"),
        (kobe.longitude_velocity).label("kobe_longitude_velocity"),
        (kix.latitude_velocity).label("kix_latitude_velocity"),
        (kix.longitude_velocity).label("kix_longitude_velocity"),
        (tomogashima.latitude_velocity).label(
            "tomogashima_latitude_velocity"),
        (tomogashima.longitude_velocity).label(
            "tomogashima_longitude_velocity"),
        (akashi.latitude_velocity).label("akashi_latitude_velocity"),
        (akashi.longitude_velocity).label("akashi_longitude_velocity"),
        (awaji.latitude_velocity).label("awaji_latitude_velocity"),
        (awaji.longitude_velocity).label("awaji_longitude_velocity"),
        (nishinomiya.latitude_velocity).label(
            "nishinomiya_latitude_velocity"),
        (nishinomiya.longitude_velocity).label(
            "nishinomiya_longitude_velocity"),
        (osaka.latitude_velocity).label("osaka_latitude_velocity"),
        (osaka.longitude_velocity).label("osaka_longitude_velocity"),
        (TemperatureTb.temperature).label("temperature"),
        (AirPressureTb.air_pressure).label("air_pressure"),
        (WaveTb.significant_height).label("height"),
        (WaveTb.significant_period).label("period")
    )\
        .join(kix, kobe.datetime == kix.datetime)\
        .join(tomogashima, kobe.datetime == tomogashima.datetime)\
        .join(akashi, kobe.datetime == akashi.datetime)\
        .join(awaji, kobe.datetime == awaji.datetime)\
        .join(nishinomiya, kobe.datetime == nishinomiya.datetime)\
        .join(osaka, kobe.datetime == osaka.datetime)\
        .join(AirPressureTb, kobe.datetime == AirPressureTb.datetime)\
        .join(WaveTb, kobe.datetime == WaveTb.datetime)\
        .join(TemperatureTb, kobe.datetime == TemperatureTb.datetime)