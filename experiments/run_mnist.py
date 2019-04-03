"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""
from layers.layer import ReluLayer, InputLayer, SoftmaxOutputLayer
from networks.network import Network
from optimizers.optimizers import SGD, SGDMomentum
import torch
import torchvision
from tensorboardX import SummaryWriter
import os

# User variables
batch_size = 128

# Initializing network

# ======== set log directory ==========
log_dir = '../logs/MNIST_BP_500n'
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

inputlayer = InputLayer(layer_dim=28 * 28, writer=writer, name='input_layer_BP')
hiddenlayer = ReluLayer(in_dim=28 * 28, layer_dim=100, writer=writer,
                        name='hidden_layer_BP')
outputlayer = SoftmaxOutputLayer(in_dim=100, layer_dim=10,
                                 loss_function='crossEntropy',
                                 name='output_layer_BP',
                                 writer=writer)

network = Network([inputlayer, hiddenlayer, outputlayer])
# if torch.cuda.is_available():
#     network.cuda(device)

# Loading dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.Compose(
                                           [
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))



train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=1000,
    shuffle=False)

# Initializing optimizer
optimizer1 = SGD(network=network, threshold=1.2, init_learning_rate=0.1,
                 tau=100,
                 final_learning_rate=0.005,
                 compute_accuracies=True, max_epoch=120)
optimizer2 = SGDMomentum(network=network, threshold=1.2, init_learning_rate=0.1,
                         tau=100, final_learning_rate=0.005,
                         compute_accuracies=True, max_epoch=150, momentum=0.5)

# Train on MNIST
optimizer1.run_mnist(train_loader, test_loader, device)

# Test network
# for batch_idx, (data,target) in enumerate(test_loader):
#     data = data.view(-1, 28 * 28, 1)
#     target = hf.one_hot(target, 10)
#     data, target = data.to(device), target.to(device)
#     predicted_classes = network.predict(data)
#     test_loss = network.loss(target)
#     test_accuracy = network.accuracy(target)
#     print('Test Loss: ' + str(test_loss))
#     print('Test Accuracy: ' + str(test_accuracy))
