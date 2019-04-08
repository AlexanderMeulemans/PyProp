from layers.layer import LinearOutputLayer
import torch
import numpy as np
from tensorboardX import SummaryWriter
import utils.helper_functions as hf
# User variables
n = 6
nb_classes = 2
batch_size = 5
h = 1e-3

writer = SummaryWriter()

layer = LinearOutputLayer(n,n,'mse',writer)
#
# targets = torch.randint(0,n,(batch_size,))
# targets = hf.one_hot(targets, n)

targets = torch.randn(batch_size,n,1)
activation = torch.randn(batch_size,n,1)
layer.forward_linear_activation = activation
layer.forward_output = activation



layer.compute_backward_output(targets)
gradient = layer.backward_output

loss = layer.loss(targets)
gradient_fd = torch.empty(activation.shape)

# compute finite difference gradient
for b in range(batch_size):
    for i in range(n):
        fd = torch.zeros(activation.shape)
        fd[b,i,0] = h
        activation_fd = activation + fd
        layer.forward_linear_activation = activation_fd
        layer.forward_output = activation_fd
        loss_fd = layer.loss(targets)
        gradient_fd[b,i,0] = (loss_fd-loss)/h

error = torch.norm(gradient-gradient_fd, p=float('inf'))

print('error: {}'.format(error))
print(gradient)
print(gradient_fd)


