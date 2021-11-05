import os
import glob
from posixpath import basename


def get_showcase_dir():
    print("please select showcase in the following")
    cases = sorted(
        list(map(lambda x: basename(x[:-1]), glob.glob("result/*/"))))
    strcases = ""
    for index, case in enumerate(cases):
        strcases += f"\n\t{index}: " + case
    print(strcases + "\n\tq: Quit" + "\n")

    while True:
        print("please type show case number")
        caseindex = input("\t:")
        try:
            caseindex = int(caseindex)
        except:
            if caseindex == "q":
                exit()
            print("\n[ error! ] case number was not typed!\n")
            continue
        if not (0 <= caseindex and len(cases) > caseindex):
            print("\n[ error! ] there is no this case number\n")
            continue
        break

    load_dir = f"result/{cases[caseindex]}/"
    if not os.path.exists(load_dir):
        assert "this path is not exists"
    return load_dir


def read_observed(load_dir: str):
    with open(load_dir+"observed.csv") as f:
        real_values = list(map(lambda x: float(x), f.read().splitlines()))
    return real_values


def read_predicted(load_dir: str):
    with open(load_dir+"predicted.csv") as f:
        pred = list(map(lambda x: float(x), f.read().splitlines()))
    return pred
