import numpy as np
import matplotlib.pyplot as plt

import utills

casedir = utills.get_showcase_dir()
print(f"\n{casedir}", end="\n\n")

for col in range(1):
    # observed = np.loadtxt(casedir+"observed.csv", delimiter=",", dtype=np.float64)[:, col]
    observed = np.loadtxt(casedir+"observed.csv", delimiter=",", dtype=np.float64)
    # pred = np.loadtxt(casedir+"predicted.csv", delimiter=",", dtype=np.float64)[:, col]
    pred = np.loadtxt(casedir+"predicted.csv", delimiter=",", dtype=np.float64)

    alldata_count = observed.shape
    mse = np.mean(np.square(observed - pred))
    std = np.sqrt(mse)

    before = observed[:-1]
    after = observed[1:]
    timeGapMSE = np.mean(np.square(before - after))

    print(
        "full data",
        f"\tcount : {alldata_count}",
        f"\tmse: {mse}", f"\tstd: {std}",
        f"\t(1 hour gap) mse: {timeGapMSE}",
        f"\t(1 hour gap) std: {np.sqrt(timeGapMSE)}",
        sep="\n", end="\n\n")

    height_threshold = 0.5
    boolarray = observed > height_threshold
    select_observed = observed[boolarray]
    select_pred = pred[boolarray]

    selectdata_count = select_observed.shape
    mse = np.mean(np.square(select_observed - select_pred))
    std = np.sqrt(mse)

    print(
        "wave height is 0.5 or more",
        f"\tcount : {selectdata_count}",
        f"\tmse: {mse}", f"\tstd: {std}",
        sep="\n", end="\n\n")

    fig = plt.figure()
    height_ax = fig.add_subplot(
        111, title=casedir+str(col), ylabel="wave height")
    height_ax.plot(range(len(observed)), observed, label="observed value")
    height_ax.plot(
        range(len(pred)), pred,
        label="predicted value", alpha=0.5, color="red")
    height_ax.grid()
    height_ax.legend()
plt.show()
