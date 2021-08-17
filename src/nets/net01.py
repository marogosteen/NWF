from torch import nn

class NNWF_Net01(nn.Module):
    def __init__(self):
        super(NNWF_Net01, self).__init__()
        self.linearSequential = nn.Sequential(
            nn.Linear(4,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1),
        )
    
    def forward(self,x):
        x = self.linearSequential(x)
        return x