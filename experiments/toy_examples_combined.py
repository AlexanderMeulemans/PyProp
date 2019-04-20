"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD, SGDInvertible
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
import random
import utils.helper_functions as hf

# good results for seed 32, no good results for seed 47
seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ======== User variables ============
nb_training_batches = 2000
batch_size = 1
testing_size = 1000
n = 3
distance = 1.5
max_epoch = 18
CPU = True
main_dir = '../logs/combined_figure_toy_example15'
tau = 30



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
# set log directory
log_dir = main_dir + '/dataset'
writer = SummaryWriter(log_dir=log_dir)

input_layer_true = InputLayer(layer_dim=n, writer=writer,
                              name='input_layer_true_model')
hidden_layer_true = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                   writer=writer,
                                   name='hidden_layer_true_model')
output_layer_true = LinearOutputLayer(in_dim=n, layer_dim=n,
                                      loss_function='mse',
                                      writer=writer,
                                      name='output_layer_true_model')
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
# ===== Run experiment with shallow BP network =======
# set log directory
log_dir = main_dir + '/shallow'
writer = SummaryWriter(log_dir=log_dir)

input_layer = InputLayer(layer_dim=n, writer=writer,
                                 name='input_layer')
hidden_layer_fixed = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                 writer=writer,
                                 name='hidden_layer',
                              fixed=True)
output_layer = LinearOutputLayer(in_dim=n, layer_dim=n,
                                         loss_function='mse',
                                         writer=writer,
                                         name='output_layer')

output_layer.set_forward_parameters(output_weights, output_layer.forward_bias)
hidden_layer_fixed.set_forward_parameters(hidden_weights, hidden_layer_fixed.forward_bias)
shallow_network = Network([input_layer, hidden_layer_fixed, output_layer])

optimizer1 = SGD(network=shallow_network, threshold=0.000001,
                 init_learning_rate=0.0015,
                 tau=tau,
                 final_learning_rate=0.0005, compute_accuracies=False,
                 max_epoch=max_epoch,
                 outputfile_name='resultfile_shallow.csv')

start_time = time.time()
optimizer1.run_dataset(input_dataset, output_dataset, input_dataset_test,
                       output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time - start_time))

# ===== Run experiment with BP network ========
log_dir = main_dir + '/BP'
writer = SummaryWriter(log_dir=log_dir)


input_layer = InputLayer(layer_dim=n, writer=writer,
                            name='input_layer')
hidden_layer = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                 writer=writer,
                                 name='hidden_layer')
output_layer = LinearOutputLayer(in_dim=n, layer_dim=n,
                                    loss_function='mse',
                                    writer=writer,
                                    name='output_layer')

hidden_layer.set_forward_parameters(hidden_weights, hidden_layer.forward_bias)
output_layer.set_forward_parameters(output_weights, output_layer.forward_bias)

BP_network = Network([input_layer, hidden_layer,
                      output_layer])

optimizer2 = SGD(network=BP_network, threshold=0.000001,
                 init_learning_rate=0.014,
                 tau=tau,
                 final_learning_rate=0.008, compute_accuracies=False,
                 max_epoch=max_epoch,
                 outputfile_name='resultfile_BP.csv')

start_time = time.time()
optimizer2.run_dataset(input_dataset, output_dataset, input_dataset_test,
                       output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time - start_time))

# ===== Run experiment with invertible TP =======
log_dir = main_dir + '/TP'
writer = SummaryWriter(log_dir=log_dir)
# Creating training network
input_layer = InvertibleInputLayer(layer_dim=n, out_dim=n, loss_function='mse',
                                  name='input_layer', writer=writer)
hidden_layer = InvertibleLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                       layer_dim=n, out_dim=n, loss_function=
                                       'mse',
                                       name='hidden_layer',
                                       writer=writer)
output_layer = InvertibleLinearOutputLayer(in_dim=n, layer_dim=n,
                                          step_size=0.01,
                                          name='output_layer',
                                          writer=writer)

hidden_layer.set_forward_parameters(hidden_weights, hidden_layer.forward_bias)
output_layer.set_forward_parameters(output_weights, output_layer.forward_bias)

network = InvertibleNetwork([input_layer, hidden_layer, output_layer])

# Initializing optimizer
optimizer3 = SGDInvertible(network=network, threshold=0.000001,
                           init_step_size=0.02, tau=tau,
                           final_step_size=0.018,
                           learning_rate=0.5, max_epoch=max_epoch)

start_time = time.time()
optimizer3.run_dataset(input_dataset, output_dataset, input_dataset_test,
                       output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time - start_time))
