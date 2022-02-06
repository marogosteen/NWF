import torch

from nwf.net import NNWF_Net

net = NNWF_Net(24)
net.load_state_dict(
    torch.load("result/5Point1HTrain1HLater2016/state_dict.pt"))
print(net.__dict__)

params = 0
count = 0
for p in net.parameters():
    if p.requires_grad:
        numel = p.numel()
        print(count, numel)
        params += numel
        count += 1

print(params)
