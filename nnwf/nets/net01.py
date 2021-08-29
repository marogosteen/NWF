from torch import nn


class NNWF_Net01(nn.Module):
    def __init__(self):
        super(NNWF_Net01, self).__init__()
        self.linearSequential = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,1),
        )
    
    def forward(self, x):
        x = self.linearSequential(x)
        return x