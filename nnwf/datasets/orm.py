import datetime

from sqlalchemy import orm, select, Column, INTEGER, REAL, TEXT
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.ext.declarative import declarative_base


# engine = create_engine("sqlite:///database/dataset.db", echo=True)
Base = declarative_base()
Datatime = DATETIME(
    storage_format="%(year)04d-%(month)02d-%(day)02d %(hour)02d:%(minute)02d",
    regexp=r"(\d+)-(\d+)-(\d+) (\d+):(\d+)")


class AmedasTb(Base):
    __tablename__ = "amedas"
    datetime = Column(Datatime, primary_key=True)
    place = Column(TEXT, primary_key=True)
    inferiority = Column(INTEGER, primary_key=True)
    latitude_velocity = Column(REAL, primary_key=True)
    longitude_velocity = Column(REAL, primary_key=True)
    temperature = Column(REAL, primary_key=True)


class NowphasTb(Base):
    __tablename__ = "nowphas"
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


kobe = orm.aliased(AmedasTb, name="kobe")
kix = orm.aliased(AmedasTb, name="kix")
tomogashima = orm.aliased(AmedasTb, name="tomogashima")


def get_sqlresult(session: orm.session.Session, begin_year: int, end_year: int):
    sqlresult = session.execute(
        select(
            (kobe.inferiority).label("kobe_inferiority"),
            (kix.inferiority).label("kix_inferiority"),
            (tomogashima.inferiority).label("tomogashima_inferiority"),
            (NowphasTb.inferiority).label("nowphas_inferiority"),
            (kobe.datetime).label("datetime"),
            (kobe.latitude_velocity).label("kobe_latitude_velocity"),
            (kobe.longitude_velocity).label("kobe_longitude_velocity"),
            (kobe.temperature).label("kobe_temperature"),
            (kix.latitude_velocity).label("kix_latitude_velocity"),
            (kix.longitude_velocity).label("kix_longitude_velocity"),
            (kix.temperature).label("kix_temperature"),
            (tomogashima.latitude_velocity).label(
                "tomogashima_latitude_velocity"),
            (tomogashima.longitude_velocity).label(
                "tomogashima_longitude_velocity"),
            (tomogashima.temperature).label("tomogashima_temperature"),
            (NowphasTb.significant_height).label("height"),
            (NowphasTb.significant_period).label("period"),
            (NowphasTb.direction).label("direction")
        )
        .join(kix, kobe.datetime == kix.datetime)
        .join(tomogashima, kobe.datetime == tomogashima.datetime)
        .join(NowphasTb, kobe.datetime == NowphasTb.datetime)
        .filter(
            kobe.place == "kobe",
            kix.place == "kix",
            tomogashima.place == "tomogashima",
            kobe.datetime >= datetime.datetime(begin_year, 1, 1),
            kobe.datetime <= datetime.datetime(end_year, 12, 31))
        .order_by(kobe.datetime))

    return sqlresult
