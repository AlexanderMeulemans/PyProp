import torch

a = torch.randn(5,5)
U,S,V = torch.svd(a)
print(S)


