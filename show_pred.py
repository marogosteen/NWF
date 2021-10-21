import matplotlib.pyplot as plt


with open("result/observed.csv") as f:
    real_values = list(map(lambda x: float(x), f.read().splitlines()))

with open("result/predicted.csv") as f:
    pred = list(map(lambda x: float(x), f.read().splitlines()))

fig = plt.figure()
height_ax = fig.add_subplot(
    111, ylabel="wave height")
height_ax.plot(range(len(real_values)), real_values, label="observed value")
height_ax.plot(
    range(len(pred)), pred,
    label="predicted value", alpha=0.5, color="red")
height_ax.grid()
height_ax.legend()
plt.show()