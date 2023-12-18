import torch
import jittor as jt

c = torch.tensor([[ 1, 2, 3], [-1, 1, 4]] , dtype=torch.float)
print(torch.norm(c, p=2))

a = jt.array([[ 1, 2, 3], [-1, 1, 4]])
b = jt.norm(a, p=2)
print(jt.norm(b, p=2))