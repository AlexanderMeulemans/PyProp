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

# ======== set log directory ==========
log_dir = '../logs/toyexample_BP'
writer = SummaryWriter(log_dir=log_dir)

# ========= Set device ===========
if torch.cuda.is_available():
    gpu_idx = 1
    device = torch.device("cuda:{}".format(gpu_idx))
    # IMPORTANT: set_default_tensor_type uses automatically device 0,
    # untill now, I did not find a fix for this
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('using GPU {}'.format(gpu_idx))
else:
    device = torch.device("cpu")
    print('using CPU')

# Create toy model dataset

input_layer_true = InputLayer(layer_dim=6, writer=writer,
                              name='input_layer_true_model')
hidden_layer_true = ReluLayer(inDim=6,layerDim=10, writer=writer,
                              name='hidden_layer_true_model')
output_layer_true = LinearOutputLayer(inDim=10, layerDim=4,
                                      lossFunction='mse',
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
print('LS train loss: '+str(train_loss))
print('LS test loss: '+str(test_loss))


# ===== Run experiment with backprop =======

# Creating training network
inputlayer = InputLayer(layer_dim=6, writer=writer, name='input_layer_BP')
hiddenlayer = ReluLayer(inDim=6, layerDim=10, writer=writer,
                                       name='hidden_layer_BP')
outputlayer = LinearOutputLayer(inDim=10, layerDim=4, lossFunction='mse',
                                          name='output_layer_BP',
                                writer=writer)

network_backprop = Network([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer4 = SGD(network=network_backprop, threshold=0.0001,
                 initLearningRate=0.01,
                 tau=100,
                 finalLearningRate=0.005, computeAccuracies=False,
                 maxEpoch=100,
                 outputfile_name='resultfile_BP.csv')



# Train on dataset
start_time = time.time()
optimizer4.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time-start_time))

# test