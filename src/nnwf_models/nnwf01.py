import torch
from torch import nn
from torch.nn.modules.activation import ReLU

class NNWF01(nn.Module):
    def __init__(self):
        super(NNWF01, self).__init__()
        self.linearSequential = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    
    def forward(self,x):
        x = self.linearSequential(x)
        return x