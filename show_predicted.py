import datetime
import json
import os

import plotly.graph_objects as go
from plotly import subplots

from nwf.report import ReportModel


def generateDatetimelist(year) -> list:
    datetimelist = []
    timeseries = datetime.datetime(year=year, month=1, day=1, hour=0, minute=0)
    duration = datetime.timedelta(hours=1)
    while timeseries.year < year + 1:
        datetimelist.append(timeseries)
        timeseries += duration
    return datetimelist


casename = "5Point2HTrain1HLater2017"
filepath = f"result/{casename}/{casename}.json"
assert os.path.exists(filepath)

reportdict = None
with open(filepath) as f:
    reportdict = json.load(f)

report1HTrain = ReportModel.__new__(ReportModel)
report1HTrain.__dict__.update(reportdict)

casename = "5Point2HTrain1HLater2017"
filepath = f"result/{casename}/{casename}.json"
assert os.path.exists(filepath)

reportdict = None
with open(filepath) as f:
    reportdict = json.load(f)

report2HTrain = ReportModel.__new__(ReportModel)
report2HTrain.__dict__.update(reportdict)

observed = report1HTrain.observed
predicted = report1HTrain.predicted
datetimelist = generateDatetimelist(report1HTrain.targetYear)

observed2H = report2HTrain.observed
predicted2H = report2HTrain.predicted

fig = subplots.make_subplots(rows=2, cols=1)

fig.add_trace(
    go.Scatter(y=observed, x=datetimelist, name="observed", marker_color="blue"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=predicted, x=datetimelist, name="predicted", marker_color="red"),
    row=1, col=1
)

fig.update_xaxes(title_text="Hour", row=1, col=1)
fig.update_yaxes(title_text="Wave Hight", row=1, col=1)

fig.add_trace(
    go.Scatter(y=observed2H, x=datetimelist, name="observed", marker_color="blue"),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=predicted2H, x=datetimelist, name="predicted", marker_color="red"),
    row=2, col=1
)

fig.update_xaxes(title_text="Hour", row=2, col=1)
fig.update_yaxes(title_text="Wave Height", row=2, col=1)

fig.show()
