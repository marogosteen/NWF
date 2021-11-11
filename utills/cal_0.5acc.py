import numpy as np

import utills


data_dir = utills.get_showcase_dir()

observed = np.array(utills.read_observed(load_dir=data_dir), dtype=np.float64)
pred = np.array(utills.read_predicted(load_dir=data_dir), dtype=np.float64)

height_threshold = 0.5
boolarray = observed > height_threshold
select_observed = observed[boolarray]
select_pred = pred[boolarray]

alldata_count = observed.shape
selectdata_count = select_observed.shape
mse = np.mean(np.square(select_observed - select_pred))
std = np.sqrt(mse)

print(
    "", data_dir, 
    f"\tcount (all data): {alldata_count}",
    f"\tcount (select data): {selectdata_count}",
    f"\tmse: {mse}", f"\tstd: {std}", 
    sep="\n")