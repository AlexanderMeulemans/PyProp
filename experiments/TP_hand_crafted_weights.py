"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD
from layers.invertible_layer import InvertibleInputLayer, \
    InvertibleLeakyReluLayer, InvertibleLinearOutputLayer
from networks.invertible_network import InvertibleNetwork
from layers.layer import InputLayer, LeakyReluLayer, \
    LinearOutputLayer
from networks.network import Network
import torch
import numpy as np
import time
from tensorboardX import SummaryWriter
from utils.LLS import linear_least_squares
import os

torch.manual_seed(47)

# ======== User variables ============
nb_training_batches = 10000
batch_size = 1
testing_size = 1000

# ======== set log directory ==========
log_dir = '../logs/TP_hand_crafted_weights'
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

# ======== Create toy model dataset =============

input_layer_true = InputLayer(layer_dim=3, writer=writer,
                              name='input_layer_true_model')
hidden_layer_true = LeakyReluLayer(negative_slope=0.35, in_dim=3, layer_dim=3,
                                   writer=writer,
                                   name='hidden_layer_true_model')
output_layer_true = LinearOutputLayer(in_dim=3, layer_dim=3,
                                      loss_function='mse',
                                      writer=writer,
                                      name='output_layer_true_model')

weights_hidden_layer = torch.Tensor([])
weights_output_layer = torch.Tensor([])
bias_hidden_layer = torch.zeros((3, 1))
bias_output_layer = torch.zeros((3, 1))

hidden_layer_true.set_forward_parameters(weights_hidden_layer,
                                         bias_hidden_layer)
output_layer_true.set_forward_parameters(weights_output_layer,
                                         bias_output_layer)

true_network = Network([input_layer_true, hidden_layer_true,
                        output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(nb_training_batches,
                                                   batch_size)
input_dataset_test, output_dataset_test = generator.generate(
    testing_size, 1)

# compute least squares solution as control
print('computing LS solution ...')
weights, train_loss, test_loss = linear_least_squares(input_dataset,
                                                      output_dataset,
                                                      input_dataset_test,
                                                      output_dataset_test)
print('LS train loss: ' + str(train_loss))
print('LS test loss: ' + str(test_loss))

# ===== Run experiment with invertible TP =======

# Creating training network
inputlayer = InvertibleInputLayer(layer_dim=5, out_dim=5, loss_function='mse',
                                  name='input_layer', writer=writer)
hiddenlayer = InvertibleLeakyReluLayer(negative_slope=0.35, in_dim=5,
                                       layer_dim=5, out_dim=5, loss_function=
                                       'mse',
                                       name='hidden_layer',
                                       writer=writer)
outputlayer = InvertibleLinearOutputLayer(in_dim=5, layer_dim=5,
                                          step_size=0.01,
                                          name='output_layer',
                                          writer=writer)

network = InvertibleNetwork([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer1 = SGD(network=network, threshold=0.001, init_learning_rate=0.01,
                 tau=100,
                 final_learning_rate=0.005, compute_accuracies=False,
                 max_epoch=120,
                 outputfile_name='resultfile.csv')

# Train on dataset
timings = np.array([])
start_time = time.time()
optimizer1.run_dataset(input_dataset, output_dataset, input_dataset_test,
                       output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time - start_time))
timings = np.append(timings, end_time - start_time)
