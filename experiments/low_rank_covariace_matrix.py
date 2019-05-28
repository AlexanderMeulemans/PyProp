import numpy as np
import matplotlib.pyplot as plt
import utils.helper_functions as hf
import torch.nn as nn
import torch
n = 100000
def forward_nonlinearity(linear_activation):
    negative_slope = 0.1
    activation_function = nn.LeakyReLU(negative_slope)
    return activation_function(linear_activation)

W = hf.get_invertible_random_matrix(5,2)
h = torch.randn((2, n))
h_W = torch.matmul(W,h)
h_r = forward_nonlinearity(h_W)

C = 1./n*torch.matmul(h,np.transpose(h))
C_W = 1./n*torch.matmul(h_W, np.transpose(h_W))
C_r = 1./n*torch.matmul(h_r, np.transpose(h_r))

U, S_W, V = torch.svd(C_W)
U, S_r, V = torch.svd(C_r)
