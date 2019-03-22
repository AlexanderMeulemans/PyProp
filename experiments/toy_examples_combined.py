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

# ======== set log directory ==========
main_dir = '../logs/toy_example_combined'
log_dir = main_dir + '/shallow_network'
writer = SummaryWriter(log_dir=log_dir)

# ======== Create toy model dataset =============

input_layer_true = InputLayer(layerDim=5, writer=writer,
                              name='input_layer_true_model')
hidden_layer_true = LeakyReluLayer(negativeSlope=0.35,inDim=5,layerDim=5,
                                   writer=writer,
                                   name='hidden_layer_true_model')
output_layer_true = LinearOutputLayer(inDim=5, layerDim=5,
                                      lossFunction='mse',
                                      writer=writer,
                                      name='output_layer_true_model')
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
print('LS train loss: '+str(train_loss))
print('LS test loss: '+str(test_loss))

# ===== Run experiment with shallow BP network =======
input_layer_shallow = InputLayer(layerDim=5, writer=writer,
                                 name='input_layer')
output_layer_shallow = LinearOutputLayer(inDim=5, layerDim=5,
                                      lossFunction='mse',
                                      writer=writer,
                                      name='output_layer')
shallow_network = Network([input_layer_shallow, output_layer_shallow])

optimizer1 = SGD(network=shallow_network,threshold=0.001, initLearningRate=0.01,
                 tau= 100,
                finalLearningRate=0.005, computeAccuracies=False,
                 maxEpoch=100,
                 outputfile_name='resultfile_shallow.csv')

start_time = time.time()
optimizer1.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time-start_time))

# ===== Run experiment with BP network ========
log_dir = main_dir + '/BP_network'
writer = SummaryWriter(log_dir=log_dir)

input_layer_BP = InputLayer(layerDim=5, writer=writer,
                              name='input_layer')
hidden_layer_BP= LeakyReluLayer(negativeSlope=0.35,inDim=5,layerDim=5,
                                   writer=writer,
                                   name='hidden_layer')
output_layer_BP = LinearOutputLayer(inDim=5, layerDim=5,
                                      lossFunction='mse',
                                      writer=writer,
                                      name='output_layer')
BP_network = Network([input_layer_BP, hidden_layer_BP,
                                  output_layer_BP])

optimizer2 = SGD(network=BP_network,threshold=0.001, initLearningRate=0.01,
                 tau= 100,
                finalLearningRate=0.005, computeAccuracies=False,
                 maxEpoch=100,
                 outputfile_name='resultfile_BP.csv')

start_time = time.time()
optimizer2.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time-start_time))

# ===== Run experiment with invertible TP =======
log_dir = main_dir + '/TP_network'
writer = SummaryWriter(log_dir=log_dir)
# Creating training network
inputlayer = InvertibleInputLayer(layerDim=5,outDim=5, lossFunction='mse',
                                  name='input_layer', writer=writer)
hiddenlayer = InvertibleLeakyReluLayer(negativeSlope=0.35, inDim=5,
                                        layerDim=5, outDim=5, lossFunction=
                                        'mse',
                                       name='hidden_layer',
                                       writer=writer)
outputlayer = InvertibleLinearOutputLayer(inDim=5, layerDim=5,
                                              stepsize=0.01,
                                          name='output_layer',
                                          writer=writer)

network = InvertibleNetwork([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer3 = SGD(network=network,threshold=0.001, initLearningRate=0.01,
                 tau= 100,
                finalLearningRate=0.005, computeAccuracies=False,
                 maxEpoch=100,
                 outputfile_name='resultfile_TP.csv')

start_time = time.time()
optimizer3.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time-start_time))
