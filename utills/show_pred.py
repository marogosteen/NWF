import matplotlib.pyplot as plt

import utills

casedir = utills.get_showcase_dir()
real_values = utills.read_observed(casedir)
pred = utills.read_predicted(casedir)

fig = plt.figure()
height_ax = fig.add_subplot(
    111, title=casedir, ylabel="wave height")
height_ax.plot(range(len(real_values)), real_values, label="observed value")
height_ax.plot(
    range(len(pred)), pred,
    label="predicted value", alpha=0.5, color="red")
height_ax.grid()
height_ax.legend()
plt.show()