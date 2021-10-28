import numpy as np

import utills


casedir = utills.get_showcase_dir()
real_values = np.array(utills.read_observed(casedir))

data1 = real_values[:-1]
data2 = real_values[1:]
mse = np.mean(np.square(data1 - data2))
print("mse: ", mse)
print("std: ", np.sqrt(mse))
