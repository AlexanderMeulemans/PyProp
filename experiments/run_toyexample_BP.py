"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD
from layers.layer import InputLayer, ReluLayer, \
    LinearOutputLayer
from networks.network import Network
import torch
import time
from tensorboardX import SummaryWriter
from utils.LLS import linear_least_squares
import os

torch.manual_seed(47)

# ======== User variables ============
nb_training_batches = 1000
batch_size = 10
testing_size = 1000
CPU = True

# ======== set log directory ==========
log_dir = '../logs/toyexample_BP'
writer = SummaryWriter(log_dir=log_dir)

# ========= Set device ===========
if not CPU:
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
else:
    device = torch.device("cpu")
    print('using CPU')

# Create toy model dataset

input_layer_true = InputLayer(layer_dim=6, writer=writer,
                              name='input_layer_true_model')
hidden_layer_true = ReluLayer(in_dim=6, layer_dim=10, writer=writer,
                              name='hidden_layer_true_model')
output_layer_true = LinearOutputLayer(in_dim=10, layer_dim=4,
                                      loss_function='mse',
                                      writer=writer,
                                      name='output_layer_true_model')
true_network = Network([input_layer_true, hidden_layer_true,
                        output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(nb_training_batches,
                                                   batch_size)
input_dataset_test, output_dataset_test = generator.generate(1,
                                                             testing_size)

# compute least squares solution as control
print('computing LS solution ...')
weights, train_loss, test_loss = linear_least_squares(input_dataset,
                                                      output_dataset,
                                                      input_dataset_test,
                                                      output_dataset_test)
print('LS train loss: ' + str(train_loss))
print('LS test loss: ' + str(test_loss))

# ===== Run experiment with backprop =======

# Creating training network
inputlayer = InputLayer(layer_dim=6, writer=writer, name='input_layer_BP')
hiddenlayer = ReluLayer(in_dim=6, layer_dim=10, writer=writer,
                        name='hidden_layer_BP')
outputlayer = LinearOutputLayer(in_dim=10, layer_dim=4, loss_function='mse',
                                name='output_layer_BP',
                                writer=writer)

network_backprop = Network([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer4 = SGD(network=network_backprop, threshold=0.0001,
                 init_learning_rate=0.01,
                 tau=100,
                 final_learning_rate=0.005, compute_accuracies=False,
                 max_epoch=100,
                 outputfile_name='resultfile_BP.csv')

# Train on dataset
start_time = time.time()
optimizer4.run_dataset(input_dataset, output_dataset, input_dataset_test,
                       output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time - start_time))

# test
