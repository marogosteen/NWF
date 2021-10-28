import numpy as np


load_dir = "result/air_pressure/"

with open(load_dir+"observed.csv") as f:
    real_values = np.array(list(map(lambda x: float(x), f.read().splitlines())))

data1 = real_values[:-1]
data2 = real_values[1:]
mse = np.mean(np.square(data1 - data2))
print("mse: ", mse)
print("std: ", np.sqrt(mse))