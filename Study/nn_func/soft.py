import torch.nn.functional as F
import torch

a = torch.tensor([[3, 3, 3]], dtype=torch.float)
# f =
print(F.log_softmax(a, dim=1))