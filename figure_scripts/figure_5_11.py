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
learning_rate_init = 0.002
learning_rate_final = 0.0001
l = 0
rows = 5
cols = 5
training_iterations = 2000
batch_sizes = [8, 8, 8]
plots = 3
fontsize = 22
legend = (r'layer size 5 $\rightarrow$ 3', r'layer size 5 $\rightarrow$ 5', r'layer size 3 $\rightarrow$ 5')
"""
asdf

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
inverse_errors = torch.zeros((plots, training_iterations))



# Nonlinearities
def forward_nonlinearity(linear_activation):
    activation_function = nn.Tanh()
    return activation_function(linear_activation)

def derivative_tanh(linear_activation):
    tanh = forward_nonlinearity(linear_activation)
    return 1 - tanh**2

# Training
for j, batch_size in enumerate(batch_sizes):
    if j == 0:
        rows = 3
        cols = 5
    elif j==1:
        rows=5
        cols=5
    else:
        rows=5
        cols=3
    forward_weights = hf.get_invertible_random_matrix(rows, cols)
    backward_weights = hf.get_invertible_random_matrix(cols, rows)
    for i in range(training_iterations):
        learning_rate = learning_rate_init + float(i/training_iterations)*(learning_rate_final - learning_rate_init)
        noise_input = torch.randn((batch_size,cols,1))
        linear_activation = torch.matmul(forward_weights,
                                         noise_input)
        nonlinear_activation = forward_nonlinearity(linear_activation)
        linear_activation2 = torch.matmul(backward_weights,
                                          nonlinear_activation)
        nonlinear_activation2 = forward_nonlinearity(linear_activation2)
        approx_error = nonlinear_activation2 - noise_input
        gradient = torch.matmul(derivative_tanh(linear_activation2) * approx_error,
                                torch.transpose(nonlinear_activation, -1, -2))

        inverse_errors[j,i] = torch.mean(torch.norm(approx_error, dim=1))

        backward_weights = (1-l*learning_rate)*backward_weights - \
                           learning_rate * torch.mean(gradient,0)


print('plotting...')


fig = plt.figure('inverse errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
for i in range(plots):
    plt.plot(inverse_errors[i,:].numpy())
plt.xlabel('iteration', fontsize=fontsize)
plt.ylabel(r'$\| g\big(f(h)\big) - h\|_{2}$', fontsize=fontsize)
plt.legend(legend, fontsize=fontsize)
plt.show()







