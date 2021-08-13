import re
import datetime

saveFilePath = "data/kobe2019/wave.csv"
readFilePath = "data/kobe2019/h306e019.txt"

with open(saveFilePath, "w") as wf:
    wf.write(
        "datetime,Significant wave height[m],Significant wave period[sec]\n"
    )
    with open(readFilePath, encoding="Shift-jis") as rf:
        count = 0
        for line in rf.readlines()[1:]:
            strDataDatetime = line[:12]
            lineData = line[11:]
            lineData = re.split(" +",lineData)

            dataDatetime = datetime.datetime(
                year=int(strDataDatetime[:4]), month=int(strDataDatetime[4:6]), day=int(strDataDatetime[6:8]),
                hour=int(strDataDatetime[8:10]), minute=int(strDataDatetime[10:12])
            )

            if dataDatetime.minute == 0:
                wf.write(dataDatetime.strftime(
                    "%Y/%m/%d/%H:%M,"+str(lineData[5])+","+str(lineData[6])+"\n")
                )

