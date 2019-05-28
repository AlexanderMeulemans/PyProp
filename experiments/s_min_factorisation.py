import torch
import utils.helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt

iterations = 1000
factors = 4
n = 5

for iteration in range(iterations):
    W_tot = torch.eye(n)
    s_tot = 1.
    for i in range(factors):
        W = hf.get_invertible_random_matrix(n,n)
        s_tot

