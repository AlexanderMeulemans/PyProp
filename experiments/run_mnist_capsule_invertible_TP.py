"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import sys
sys.path.append('.')
from layers.invertible_layer import InvertibleLeakyReluLayer
from layers.invertible_layer import InvertibleInputLayer
from layers.invertible_layer import InvertibleSoftmaxOutputLayer
from networks.invertible_network import InvertibleNetwork
from optimizers.optimizers import SGD, SGDMomentum, SGDInvertible
import torch
import torchvision
from tensorboardX import SummaryWriter
import os
import random
import numpy as np

seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# User variables
batch_size = 1
n = 28*28
negative_slope = 0.1
debug_mode = False

# Initializing network

# ======== set log directory ==========
log_dir = '../logs/MNIST_TP_capsule'
writer = SummaryWriter(log_dir=log_dir)

# ======== set device ============
if torch.cuda.is_available():
    gpu_idx = 0
    device = torch.device("cuda:{}".format(gpu_idx))
    # IMPORTANT: set_default_tensor_type uses automatically device 0,
    # untill now, I did not find a fix for this
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('using GPU')
else:
    device = torch.device("cpu")
    print('using CPU')

# ======== Design network =============

input_layer = InvertibleInputLayer(
    layer_dim=28 * 28,
    out_dim=100,
    writer=writer,
    name='input_layer',
    debug_mode=debug_mode
)
hidden_layer = InvertibleLeakyReluLayer(
    negative_slope=negative_slope,
    in_dim=28 * 28,
    layer_dim=100,
    out_dim=10,
    writer=writer,
    name='hidden_layer',
    debug_mode=debug_mode
)
output_layer = InvertibleSoftmaxOutputLayer(
    in_dim=100,
    layer_dim=10,
    writer=writer,
    step_size=0.01,
    name='output_layer',
    debug_mode=debug_mode
)


network = InvertibleNetwork([input_layer, hidden_layer, output_layer])


# Loading dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.Compose(
                                           [
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,),
                                                   (0.3081,))
                                           ]))
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,),
                                              (0.3081,))
                                      ]))



train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

# Initializing optimizer
optimizer1 = SGD(network=network, threshold=0.0001, init_learning_rate=0.1,
                 tau=100,
                 final_learning_rate=0.005,
                 compute_accuracies=True, max_epoch=120)
optimizer2 = SGDMomentum(network=network, threshold=0.0001, init_learning_rate=1.0,
                         tau=100, final_learning_rate=0.05,
                         compute_accuracies=True, max_epoch=150, momentum=0.5)
optimizer3 = SGDInvertible(
    network=network,
    threshold=0.001,
    init_step_size=0.2,
    tau=200,
    final_step_size=0.01,
    learning_rate=0.5,
    compute_accuracies=True,
    max_epoch=250
)

# Train on MNIST
optimizer3.run_mnist(train_loader, test_loader, device)