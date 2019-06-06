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

# User variables

learning_rate_init = 0.006
learning_rate_final = 0.001
l_init = 0
l = l_init
rows = 5
cols = 5
training_iterations = 2500
batch_sizes = [1, 1, 1]
plots = 3
legend = ('white input', 'coloured full rank', 'coloured low rank')
fontsize = 22
"""
53 setting: 
learning_rate_init = 0.02
learning_rate_final = 0.008
l = 2e-1
rows = 5
cols = 3
training_iterations = 2000

53 no LM setting:
learning_rate_init = 0.02
learning_rate_final = 0.008
l = 0
rows = 5
cols = 3
training_iterations = 600

35 coloured setting:
learning_rate_init = 0.02
learning_rate_final = 0.0001
l = 0
rows = 3
cols = 5
training_iterations = 2000
batch_sizes = [1, 16, 16]
plots = 3
fontsize = 23

53 combined setting:
learning_rate_init = 0.02
learning_rate_final = 0.008
l_init = 2e-1
l = l_init
rows = 5
cols = 3
training_iterations = 2000
batch_sizes = [1, 1, 1]
plots = 3
legend = ('unregularized', 'regularized', 'coloured input')
fontsize = 22

55 combined setting:
learning_rate_init = 0.01
learning_rate_final = 0.008
l_init = 0
l = l_init
rows = 5
cols = 5
training_iterations = 2500
batch_sizes = [1, 1, 1]
plots = 3
legend = ('white input', 'coloured full rank', 'coloured low rank')
fontsize = 22

35 combined setting:
learning_rate_init = 0.02
learning_rate_final = 0.0001
l = 0
rows = 3
cols = 5
training_iterations = 2000
batch_sizes = [1, 1, 16]
plots = 3
fontsize = 22
legend = ('coloured input', 'batch size = 1', 'batch size = 16')

"""

# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Forward parameters
forward_weights = hf.get_invertible_random_matrix(rows, cols)
forward_weights_pinv = torch.pinverse(forward_weights)
forward_weights_LM = torch.matmul(torch.transpose(forward_weights, 0,1),
                torch.inverse(torch.matmul(forward_weights,
                                           torch.transpose(forward_weights, 0,1)) + \
                    l*torch.eye(rows)))
random_matrix = torch.randn((cols,cols))
random_matrix2 = torch.randn((cols, 2))

# Backward parameters
backward_weights_init = hf.get_invertible_random_matrix(cols, rows)

# Result arrays
matrix_errors = torch.zeros((plots, training_iterations))
matrix_errors_LM = torch.zeros((plots, training_iterations))
inverse_errors = torch.zeros((plots, training_iterations))



# Nonlinearities
def forward_nonlinearity(linear_activation):
    negative_slope = 0.35
    activation_function = nn.LeakyReLU(negative_slope)
    return activation_function(linear_activation)

def backward_nonlinearity(input):
    negative_slope = 0.35
    output = torch.empty(input.shape)
    for i in range(input.size(0)):
        for j in range(input.size(1)):
            for k in range(input.size(2)):
                if input[i, j] >= 0:
                    output[i, j] = input[i, j,k]
                else:
                    output[i, j] = input[i, j,k] / negative_slope
    return output

# Training
for j, batch_size in enumerate(batch_sizes):
    backward_weights = backward_weights_init
    for i in range(training_iterations):
        learning_rate = learning_rate_init + float(i/training_iterations)*(learning_rate_final - learning_rate_init)
        noise_input = torch.randn((batch_size,cols,1))
        if j == 2:
            noise_input = torch.matmul(random_matrix2, torch.randn((batch_size, 2,1)))
        if j == 0:
            noise_input = torch.matmul(random_matrix, noise_input)
        linear_activation = torch.matmul(forward_weights,
                                         noise_input)
        nonlinear_activation = forward_nonlinearity(linear_activation)
        nonlinear_activation2 = backward_nonlinearity(nonlinear_activation)
        linear_activation2 = torch.matmul(backward_weights,
                                          nonlinear_activation2)
        approx_error = linear_activation2 - noise_input
        gradient = torch.matmul(approx_error,
                                torch.transpose(nonlinear_activation2, -1, -2))

        matrix_errors[j,i] = torch.norm(backward_weights - forward_weights_pinv,
                                      p=float('inf'))
        matrix_errors_LM[j,i] = torch.norm(backward_weights - forward_weights_LM,
                                      p=float('inf'))
        inverse_errors[j,i] = torch.mean(torch.norm(approx_error, dim=1))

        backward_weights = (1-l*learning_rate)*backward_weights - \
                           learning_rate * torch.mean(gradient,0)


print('plotting...')
fig = plt.figure('matrix errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
for i in range(plots):
    plt.plot(matrix_errors[i,:].numpy())
plt.xlabel('iteration', fontsize=fontsize)
plt.ylabel(r'$\| Q - W^{\dagger}\|_{\infty}$', fontsize=fontsize)
plt.legend(legend, fontsize=fontsize)
plt.show()

fig = plt.figure('inverse errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
for i in range(plots):
    plt.plot(inverse_errors[i,:].numpy())
plt.xlabel('iteration', fontsize=fontsize)
plt.ylabel(r'$\| g\big(f(h)\big) - h\|_{2}$', fontsize=fontsize)
plt.legend(legend, fontsize=fontsize)
plt.show()

fig = plt.figure('LM matrix errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
for i in range(plots):
    plt.plot(matrix_errors_LM[i,:].numpy())
plt.xlabel('iteration', fontsize=fontsize)
plt.ylabel(r'$\| Q - W^{LM}\|_{\infty}$', fontsize=fontsize)
plt.legend(legend, fontsize=fontsize)
plt.show()

# 35 coloured setting:
# print('plotting...')
# fig = plt.figure('matrix errors')
# ax = fig.add_subplot(1, 1, 1)
# ax.tick_params(axis='both', which='major', labelsize=21)
# for i in range(plots):
#     plt.plot(matrix_errors[i,:].numpy())
# plt.xlabel('iteration', fontsize=fontsize)
# plt.ylabel(r'$\| Q - W^{\dagger}\|_{\infty}$', fontsize=fontsize)
# plt.legend(('batch size = 1', 'batch size = 16', 'coloured input'), fontsize=fontsize)
# plt.show()
#
# fig = plt.figure('inverse errors')
# ax = fig.add_subplot(1, 1, 1)
# ax.tick_params(axis='both', which='major', labelsize=21)
# for i in range(plots):
#     plt.plot(inverse_errors[i,:].numpy())
# plt.xlabel('iteration', fontsize=fontsize)
# plt.ylabel(r'$\| g\big(f(h)\big) - h\|_{2}$', fontsize=fontsize)
# plt.legend(('batch size = 1', 'batch size = 16', 'coloured input'), fontsize=fontsize)
# plt.show()
#
# fig = plt.figure('LM matrix errors')
# ax = fig.add_subplot(1, 1, 1)
# ax.tick_params(axis='both', which='major', labelsize=21)
# for i in range(plots):
#     plt.plot(matrix_errors_LM[i,:].numpy())
# plt.xlabel('iteration', fontsize=fontsize)
# plt.ylabel(r'$\| Q - W^{LM}\|_{\infty}$', fontsize=fontsize)
# plt.legend(('batch size = 1', 'batch size = 16', 'coloured input'), fontsize=fontsize)
# plt.show()








