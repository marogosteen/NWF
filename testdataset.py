from nnwf.dataset import TrainDatasetModel as Tds, EvalDatasetModel as Eds

with Eds(forecast_hour=1, train_hour=2, targetyear=2016) as train_dataset:
    datetimeList = train_dataset.datetimeList()
    inferiorityList = train_dataset.inferiorityList()
    print("datetime count", len(datetimeList))
    print("inferiority count", len(inferiorityList))
    print("inferiority false", inferiorityList.count(False))
    print("inferiority true", inferiorityList.count(True))
    print("dataset count", len(train_dataset))
    for record in train_dataset.recordBuffer:
        print("record buffer", record)

