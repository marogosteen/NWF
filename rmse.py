import json
import os
import datetime

import matplotlib.pyplot as plt
from matplotlib import dates
import japanize_matplotlib
import numpy as np

from nwf.report import ReportModel


def calcrmse(observed, predicted):
    observed = observed[observed != None]
    predicted = predicted[predicted != None]
    error = observed - predicted
    mse = np.mean(np.square(error))
    rmse = np.sqrt(mse)

    height_threshold = 0.5
    boolarray = observed > height_threshold
    select_observed = observed[boolarray]
    select_predicted = predicted[boolarray]

    error = select_observed - select_predicted
    mse = np.mean(np.square(error))
    rmse05 = np.sqrt(mse)

    return mse, rmse, rmse05


def generateDatetimelist(year):
    datetimelist = []
    timeseries = datetime.datetime(year=year, month=1, day=1, hour=0, minute=0)
    duration = datetime.timedelta(hours=1)
    while timeseries.year < year + 1:
        datetimelist.append(timeseries)
        timeseries += duration
    return datetimelist


# def drawWaveHeight(savepath, observed, predicted, datetimelist):
#     plt.rcParams["font.size"] = 17
#     begindatetime = datetimelist[0]
#     enddatetime = datetimelist[-1]

#     fig = plt.figure(figsize=(7, 5))
#     ax = fig.add_subplot(111, ylabel="有義波高 [m]", xlabel="日付",
#                          ylim=[0, 5], xlim=[begindatetime, enddatetime])
#     ax.plot(datetimelist, observed, label="観測値")
#     ax.plot(datetimelist, predicted, label="予測値")
#     plt.xticks(rotation=90)
#     plt.grid()
#     plt.legend()
#     plt.tight_layout()

#     plt.savefig(savepath)
#     plt.close()


def drawJebi(observed, predicted, datetimelist):
    plt.rcParams["font.size"] = 15
    begindatetime = datetime.datetime(
        year=2018, month=8, day=23, hour=0, minute=0)
    enddatetime = datetime.datetime(
        year=2018, month=9, day=6, hour=0, minute=0)
    currentdatetime = begindatetime
    duration = datetime.timedelta(days=2)
    ticks = []
    while currentdatetime <= enddatetime:
        ticks.append(currentdatetime)
        currentdatetime += duration
    print(ticks)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, ylabel="有義波高 [m]", xlabel="日付",
                         ylim=[0, 5], xlim=[begindatetime, enddatetime])
    ax.plot(datetimelist, observed, label="観測値")
    ax.plot(datetimelist, predicted, label="予測値")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y年\n%m月%d日%H時'))

    plt.xticks(rotation=90)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig("Jebi.jpg")
    plt.close()


trainHour = 1
for forecastHour in range(1, 6)[0:1]:
    for year in range(2016, 2020):
        print(f"\n\n{year}")

        jsonpath = f"result/5Point{trainHour}HTrain{forecastHour}HLater{year}/5Point{trainHour}HTrain{forecastHour}HLater{year}.json"
        reportdict = None
        with open(jsonpath) as f:
            reportdict = json.load(f)

        report = ReportModel.__new__(ReportModel)
        report.__dict__.update(reportdict)
        observed = np.array(report.observed)
        predicted = np.array(report.predicted)

        casename = os.path.basename(os.path.dirname(jsonpath))
        datetimelist = generateDatetimelist(year)
        # drawWaveHeight(casename + ".jpg", observed, predicted, datetimelist)

        if year == 2018:
            drawJebi(observed, predicted, datetimelist)

        mse, rmse, rmse05 = calcrmse(observed, predicted)
        print(
            f"mse: {round(mse, 7)}",
            f"rmse: {round(rmse, 7)}",
            f"rmse0.5: {round(rmse05, 7)}",
            sep="\n")

        before = observed[:-1]
        after = observed[1:]
        newbefore = np.where(before == None, None, after)
        newafter = np.where(after == None, None, before)
        before = newbefore
        after = newafter

        mse, rmse, rmse05 = calcrmse(after, before)
        print(
            f"1HGap mse: {round(mse, 7)}",
            f"1HGap rmse: {round(rmse, 7)}",
            f"1HGap rmse0.5: {round(rmse05, 7)}",
            sep="\n")
