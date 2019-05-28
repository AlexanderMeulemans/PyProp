"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD, SGDInvertible
from layers.target_prop_layer import TargetPropInputLayer, \
    TargetPropLeakyReluLayer, TargetPropLinearOutputLayer
from layers.DTP_layer import DTPLinearOutputLayer, DTPInputLayer, \
    DTPLeakyReluLayer
from layers.original_TP_layer import OriginalTPLinearOutputLayer, \
    OriginalTPLeakyReluLayer, OriginalTPInputLayer
from layers.original_DTP_layer import OriginalDTPLinearOutputLayer, \
    OriginalDTPLeakyReluLayer, OriginalDTPInputLayer
from networks.target_prop_network import TargetPropNetwork
from layers.layer import InputLayer, LeakyReluLayer, \
    LinearOutputLayer
from networks.network import Network
import torch
import numpy as np
import time
from tensorboardX import SummaryWriter
from utils.LLS import linear_least_squares
import os
import random
import utils.helper_functions as hf
import matplotlib.pyplot as plt

# good results for seed 32, no good results for seed 47
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ======== User variables ============
nb_training_batches = 120
batch_size = 32
testing_size = 2
n = 3
distance = 5.
CPU = True
debug = False
weight_decay = 0.0000
learning_rate_TP = 0.1
learning_rate_BP = 0.05
learning_rate_DTP = 0.1
learning_rate_fixed = 0.05
learning_rate_originalTP = 0.05
learning_rate_originalDTP = 0.1
weight_decay_TP = 0.0
weight_decay_DTP = 0.0
weight_decay_originalTP = 0.0
weight_decay_originalDTP = 0.0

output_step_size = 0.1
randomize = True
max_epoch = 60
logs=True
threshold = 0.0000001
# ======== set log directory ==========
log_dir_main = '../logs/final_combined_toy_example/'

# ======== set device ============
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

# ======== Create toy model dataset =============
log_dir = log_dir_main + 'dataset'
writer = SummaryWriter(log_dir=log_dir)
input_layer_true = InputLayer(layer_dim=n, writer=writer,
                              name='input_layer_true_model',
                              debug_mode=debug,
                              weight_decay=weight_decay)
hidden_layer_true = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                   writer=writer,
                                   name='hidden_layer_true_model',
                                   debug_mode=debug,
                                   weight_decay=weight_decay)
output_layer_true = LinearOutputLayer(in_dim=n, layer_dim=n,
                                      loss_function='mse',
                                      writer=writer,
                                      name='output_layer_true_model',
                                      debug_mode=debug,
                                      weight_decay=weight_decay)
true_network = Network([input_layer_true, hidden_layer_true,
                        output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(nb_training_batches,
                                                   batch_size)
input_dataset_test, output_dataset_test = generator.generate(
    testing_size, 1)

output_weights_true = output_layer_true.forward_weights
hidden_weights_true = hidden_layer_true.forward_weights

output_weights = hf.get_invertible_neighbourhood_matrix(output_weights_true,
                                                        distance)
hidden_weights = hf.get_invertible_neighbourhood_matrix(hidden_weights_true,
                                                        distance)


# compute least squares solution as control
print('computing LS solution ...')
weights, train_loss, test_loss = linear_least_squares(input_dataset,
                                                      output_dataset,
                                                      input_dataset_test,
                                                      output_dataset_test)
print('LS train loss: ' + str(train_loss))
print('LS test loss: ' + str(test_loss))

# ===== Run experiment with BP =======
# initialize forward weights in the neighbourhood of true weights
log_dir = log_dir_main + 'BP'
writer = SummaryWriter(log_dir=log_dir)


# Creating training network
inputlayer_BP = InputLayer(layer_dim=n, writer=writer,
                          name='input_layer',
                          debug_mode=debug,
                          weight_decay=weight_decay)
hiddenlayer_BP = LeakyReluLayer(negative_slope=0.35, in_dim=n,
                                   layer_dim=n,
                                   writer=writer,
                                   name='hidden_layer',
                                   debug_mode=debug,
                                   weight_decay=weight_decay,
                                fixed=True)
outputlayer_BP = LinearOutputLayer(in_dim=n, layer_dim=n,
                                      loss_function='mse',
                                      writer=writer,
                                      name='output_layer',
                                      debug_mode=debug,
                                      weight_decay=weight_decay)
hiddenlayer_BP.set_forward_parameters(hidden_weights, hiddenlayer_BP.forward_bias)
outputlayer_BP.set_forward_parameters(output_weights, outputlayer_BP.forward_bias)

network_BP = Network([inputlayer_BP,
                             hiddenlayer_BP, outputlayer_BP], log=logs)

# Initializing optimizer
optimizer1 = SGD(network=network_BP, threshold=threshold,
                 init_learning_rate=learning_rate_BP,
                 tau=100,
                 final_learning_rate=learning_rate_BP/5.,
                 compute_accuracies=False,
                 max_epoch=max_epoch,
                 outputfile_name='resultfile.csv')

# Train on dataset
start_time = time.time()
train_loss_BP, test_loss_BP = optimizer1.run_dataset(
    input_dataset, output_dataset, input_dataset_test,
                       output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time - start_time))