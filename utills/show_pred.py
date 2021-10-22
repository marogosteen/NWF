import matplotlib.pyplot as plt


load_dir = "result/air_pressure/"

with open(load_dir+"observed.csv") as f:
    real_values = list(map(lambda x: float(x), f.read().splitlines()))

with open(load_dir+"predicted.csv") as f:
    pred = list(map(lambda x: float(x), f.read().splitlines()))

fig = plt.figure()
height_ax = fig.add_subplot(
    111, title=load_dir, ylabel="wave height")
height_ax.plot(range(len(real_values)), real_values, label="observed value")
height_ax.plot(
    range(len(pred)), pred,
    label="predicted value", alpha=0.5, color="red")
height_ax.grid()
height_ax.legend()
plt.show()