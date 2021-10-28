# TODO Resultの0.5以上の波に対して

import numpy as np
import matplotlib.pyplot as plt

import utills


casedir = utills.get_showcase_dir()
real_values = np.array(utills.read_observed(casedir))
pred = np.array(utills.read_predicted(casedir))

boolarray = real_values > 0.5
real_values = real_values[boolarray]
pred = pred[boolarray]


fig = plt.figure()
height_ax = fig.add_subplot(
    111, title=casedir, ylabel="observed value", xlabel="predicted value",
    xlim=(0.5, 2), ylim=(0.5, 2))
height_ax.scatter(pred, real_values)
# height_ax.plot(range(len(real_values)), real_values, label="observed value")
# height_ax.plot(
#     range(len(pred)), pred,
#     label="predicted value", alpha=0.5, color="red")
height_ax.grid()
height_ax.legend()
plt.show()
