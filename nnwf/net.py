from torch import nn


class NNWF_Net(nn.Module):
    def __init__(self, inputSize, forecastHour):
        super(NNWF_Net, self).__init__()
        self.linearSequential = nn.Sequential(
            nn.Linear(inputSize, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, forecastHour),
        )

    def forward(self, x):
        x = self.linearSequential(x)
        return x
