import torch


class BaseFoo():
    def __init__(self):
        self.count = 0

    def __iter__(self):
        self.len = 4
        self.__iterobj = [1, 2, 3, 4]
        return self

    def __next__(self):
        if self.len == self.count:
            raise StopIteration
        self.count += 1
        return self.__iterobj[self.count - 1]


class Foo(BaseFoo):
    def __init__(self):
        super().__init__()

    def __next__(self):
        return super().__next__() + 10


class Bar(Foo):
    def __init__(self):
        super().__init__()

    def __next__(self):
        return super().__next__() + 10


foo = Bar()

for item in foo:
    print(item)


def classConvert(label):
    classTensor = torch.arange(0.25, 2.25, 0.25)
    print(classTensor)
    print(classTensor - label)
    index = torch.argmin(torch.abs(classTensor - label))
    return classTensor[index]
