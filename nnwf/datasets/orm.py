import datetime

from sqlalchemy import create_engine
from sqlalchemy import orm, select, Column, INTEGER, REAL, TEXT, types
from sqlalchemy.orm import Bundle
from sqlalchemy.ext.declarative import declarative_base


engine = create_engine("sqlite:///database/dataset.db", echo=True)
Base = declarative_base()


class MyDateTime(types.TypeDecorator):
    impl = types.INTEGER
    strformat = "%Y%m%d%H%M"

    def process_bind_param(self, value, dialect):
        return int(datetime.datetime.strftime(value, self.strformat))

    def process_result_value(self, value, dialect):
        return datetime.datetime.strptime(str(value), self.strformat)


class AmedasTb(Base):
    __tablename__ = "amedas"
    datetime = Column(MyDateTime, primary_key=True)
    place = Column(TEXT, primary_key=True)
    inferiority = Column(INTEGER, primary_key=True)
    latitude_velocity = Column(REAL, primary_key=True)
    longitude_velocity = Column(REAL, primary_key=True)
    temperature = Column(REAL, primary_key=True)


class NowphasTb(Base):
    __tablename__ = "nowphas"
    datetime = Column(MyDateTime, primary_key=True)
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


def get_sqlresult(session: orm.session.Session):
    kobe = orm.aliased(AmedasTb, name="kobe")
    kix = orm.aliased(AmedasTb, name="kix")
    tomogashima = orm.aliased(AmedasTb, name="tomogashima")

    sqlresult = session.execute(
        select(
            kobe, kix, tomogashima,
            Bundle(
                "nowphas", NowphasTb.inferiority, NowphasTb.direction,
                NowphasTb.significant_height, NowphasTb.significant_period))
        .join(kix, kobe.datetime == kix.datetime)
        .join(tomogashima, kobe.datetime == tomogashima.datetime)
        .join(NowphasTb, kobe.datetime == NowphasTb.datetime)
        .filter(
            kobe.place == "kobe",
            kix.place == "kix",
            tomogashima.place == "tomogashima",
            kobe.datetime >= datetime.datetime(2016, 1, 1),
            kobe.datetime <= datetime.datetime(2019, 12, 31))
        .order_by(kobe.datetime))

    return sqlresult


def is_inferiority(self):
    inferiority = True in [
        row.kobe.inferiority, row.kix.inferiority,
        row.tomogashima.inferiority, row.nowphas.inferiority]
    return inferiority


print("\nrun\n")

Session = orm.sessionmaker(engine)
with Session() as session:
    sqlresult = get_sqlresult(session)
    print(type(sqlresult))
    rows = sqlresult.fetchmany(10)
    print(f"\nlen:{len(rows)}\n")
    for row in rows:
        print("keys", row.keys())
        print("nowphas", row.nowphas.keys())
        print(type(row.kobe.datetime), row.kobe.datetime, row.kobe.place)
        exit()

# Session = orm.sessionmaker(engine)
# with Session() as session:
#     session: SessionClass
#     kobe = orm.aliased(AmedasTb, name="kobe")
#     kix = orm.aliased(AmedasTb, name="kix")
#     tomogashima = orm.aliased(AmedasTb, name="tomogashima")

#     sqlresult = session.execute(
#         select(kobe, kix)
#         .join(kix, kobe.datetime == kix.datetime)
#         .join(tomogashima, kobe.datetime == tomogashima.datetime)
#         .join(NowphasTb, kobe.datetime == NowphasTb.datetime)
#         .filter(
#             kobe.place == "kobe",
#             kix.place == "kix",
#             tomogashima.place == "tomogashima",
#             kobe.datetime >= datetime.datetime(2019, 1, 1))
#         .order_by(kobe.datetime))

#     rows = sqlresult.fetchmany(10)
#     print(f"\nlen:{len(rows)}\n")
#     for row in rows:
#         print(type(row.kobe.datetime), row.kobe.datetime, row.kobe.place)
#         exit()


print("Done!")
