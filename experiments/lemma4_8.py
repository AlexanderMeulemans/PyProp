"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import utils.helper_functions as hf
import torch.nn as nn
import random

seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# User variables
layer_sizes = [5,5,5]
iterations = 1000
batch_size = 8
font_size = 26
rcond = 1e-12

# result arrays
matrix_errors = np.zeros(iterations)
inverse_errors = np.zeros(iterations)
inverse_errors_control = np.zeros(iterations)
h1s = np.zeros(iterations)
h2s = np.zeros(iterations)
hs = np.zeros(iterations)
s_min_reals = np.zeros(iterations)
s_min_differences = np.zeros(iterations)


for iteration in range(iterations):
    # Create weight and diagonal matrices
    weight_matrices = dict()
    diagonal_matrices = dict()
    s_mins = np.empty(len(layer_sizes)-1)
    for i in range(1, len(layer_sizes)):
        weight_matrices[i] = hf.get_invertible_random_matrix(layer_sizes[i],
                                                             layer_sizes[i-1])
        U,S,V = torch.svd(weight_matrices[i])
        s_mins[i-1] = torch.min(S)
        diagonal_matrices[i] = hf.get_invertible_diagonal_matrix(layer_sizes[i])

    J_tot = torch.eye(layer_sizes[-1])
    W_tot = torch.eye(layer_sizes[-1])
    s_min_tot = 1.
    for i in range(len(layer_sizes)-1,0,-1):
        J_tot = torch.matmul(J_tot, torch.matmul(diagonal_matrices[i],
                                                 weight_matrices[i]))
        W_tot = torch.matmul(W_tot, weight_matrices[i])
        s_min_tot = s_min_tot * s_mins[i-1]
    J_tot_pinv = torch.pinverse(J_tot, rcond=rcond)
    U,S,V = torch.svd(W_tot)
    s_min_real = torch.min(S)
    s_min_reals[iteration] = s_min_real
    s_min_difference = s_min_real - s_min_tot
    s_min_differences[iteration] = s_min_difference

    J_tot_pinv_factored = torch.eye(layer_sizes[0])
    for i in range(1,len(layer_sizes)):
        W_pinv = torch.pinverse(weight_matrices[i], rcond=rcond)
        D_pinv = torch.pinverse(diagonal_matrices[i], rcond=rcond)
        J_tot_pinv_factored = torch.matmul(J_tot_pinv_factored, torch.matmul(
            W_pinv,D_pinv
        ))
    matrix_errors[iteration] = torch.norm(J_tot_pinv - J_tot_pinv_factored,
                                      p=float('inf'))
    h = torch.randn(batch_size,layer_sizes[-1],1)
    h1 = torch.matmul(J_tot_pinv, h)
    h2 = torch.matmul(J_tot_pinv_factored, h)
    hs[iteration] = torch.norm(h)
    h1s[iteration] = torch.norm(h1)
    h2s[iteration] = torch.norm(h2)
    h_control = torch.randn(batch_size,layer_sizes[-1],1)
    inverse_errors[iteration] = torch.mean(torch.norm(h1 - h2, dim=1))
    inverse_errors_control[iteration] = torch.mean(
        torch.norm(h - h_control, dim=1))


fig = plt.figure('matrix error')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(matrix_errors,bins=100)
plt.xlabel(r'$\| J_{tot}^{\dagger} - J_{tot}^{f}\|_{\infty}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()

fig = plt.figure('inverse error')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(inverse_errors,bins=100)
plt.xlabel(r'$\| J_{tot}^{\dagger}e - J_{tot}^{f}e\|_{2}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()

fig = plt.figure('inverse error control')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(inverse_errors_control,bins=100)
plt.xlabel(r'$\| e_1 - e_2\|_{2}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()

fig = plt.figure('norm h1')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(h1s, bins=100)
plt.xlabel(r'$\| e_1\|_{2}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()

fig = plt.figure('norm h2')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(h2s, bins=100)
plt.xlabel(r'$\|e_2\|_{2}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()

fig = plt.figure('norm h')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(hs, bins=100)
plt.xlabel(r'$\| e\|_{2}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()

fig = plt.figure('s_min')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(s_min_reals, bins=100)
plt.xlabel(r'$\| e\|_{2}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()

fig = plt.figure('s_min_diff')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(s_min_differences, bins=100)
plt.xlabel(r'$\| h\|_{2}$', fontsize=font_size)
plt.ylabel(r'\# samples', fontsize=font_size)
plt.show()