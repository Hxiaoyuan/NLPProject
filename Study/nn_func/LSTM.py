import torch.nn as nn
import torch

lstm = nn.LSTM(12, 6, 1, bidirectional=True)

a = torch.rand((2, 4, 12))

c_o, c_h = lstm(a)

print(c_h)