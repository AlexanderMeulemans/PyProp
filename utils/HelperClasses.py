import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

class NetworkError(Exception):
    pass

class NotImplementedError(Exception):
    pass