import json

import numpy as np

for i in range(2, 6):
    for casename in ["HourLater2016", "HourLater2017", "HourLater2018", "HourLater2019"]:
        casename = str(i) + casename
        path = f"result/{casename}/{casename}.json"
        print(path)
        with open(path) as f:
            d = json.loads(f.read())
        d["observed"] = np.array(d["observed"])[:, 0].tolist()
        d["predicted"] = np.array(d["predicted"])[:, 0].tolist()

        with open(path, "w") as f:
            json.dump(d, f)    
