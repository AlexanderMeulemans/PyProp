
"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""
import sys
sys.path.append('.')
from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD, SGDInvertible, SGDbidirectional
from layers.original_DTP_layer import OriginalDTPInputLayer, \
    OriginalDTPLeakyReluLayer, OriginalDTPLinearOutputLayer
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
import traceback


seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ======== User variables ============
nb_training_batches = 120
batch_size = 32
testing_size = 1000
n = 6
# distances = [0.1, 0.5, 1.5, 5., 10.]
distances = [8.]
# learning_rates = [5., 1., 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
learning_rates = [0.5,0.1, 0.05, 0.01, 0.005, 0.001]
backward_weight_decay = 0.0
backward_learning_rates = [0.08, 0.05, 0.01]
output_step_size = 0.1
CPU = True
debug = False
weight_decay = 0.
randomizes = [True, False]
max_epochs = 30
logs = False
threshold = 0.00001

# ======== set log directory ==========
log_dir = '../logs/gridsearch_original_DTP_onelayer2'
writer = SummaryWriter(log_dir=log_dir)

# ======== Create result files ========7
results_train = np.zeros((len(randomizes), len(distances), len(learning_rates),
                          len(backward_learning_rates), max_epochs+1))
results_test = np.zeros((len(randomizes), len(distances), len(learning_rates),
                          len(backward_learning_rates), max_epochs+1))
succesful_run = np.ones((len(randomizes), len(distances), len(learning_rates),
                         len(backward_learning_rates)),
                         dtype=bool)
best_results = np.zeros((len(randomizes), len(distances), len(learning_rates),
                         len(backward_learning_rates)))

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

# ======= Start grid search ============
for i,randomize in enumerate(randomizes):
    for j,distance in enumerate(distances):
        output_weights = hf.get_invertible_neighbourhood_matrix(
            output_weights_true,
            distance)
        hidden_weights = hf.get_invertible_neighbourhood_matrix(
            hidden_weights_true,
            distance)
        for k, learning_rate in enumerate(learning_rates):
            for l, backward_learning_rate in enumerate(backward_learning_rates):

                print('#################################')
                print('Training combination: randomize={}, '
                      'distance={}, learning_rate={}, '
                      'backward_learning_rate={} ...'.format(randomize,
                                                             distance,
                                                             learning_rate,
                                                             backward_learning_rate))


                inputlayer = OriginalDTPInputLayer(layer_dim=n, out_dim=n,
                                                  loss_function='mse',
                                                  name='input_layer', writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay,
                                                  weight_decay_backward=backward_weight_decay)
                hiddenlayer = OriginalDTPLeakyReluLayer(negative_slope=0.35,
                                                       in_dim=n,
                                                       layer_dim=n, out_dim=n,
                                                       loss_function=
                                                       'mse',
                                                       name='hidden_layer',
                                                       writer=writer,
                                                       debug_mode=debug,
                                                       weight_decay=weight_decay,
                                                       weight_decay_backward=backward_weight_decay)
                outputlayer = OriginalDTPLinearOutputLayer(in_dim=n, layer_dim=n,
                                                          step_size=output_step_size,
                                                          name='output_layer',
                                                          writer=writer,
                                                          debug_mode=debug,
                                                          weight_decay=weight_decay)
                hiddenlayer.set_forward_parameters(hidden_weights,
                                                   hiddenlayer.forward_bias)
                outputlayer.set_forward_parameters(output_weights,
                                                   outputlayer.forward_bias)

                network = TargetPropNetwork([inputlayer, hiddenlayer, outputlayer],
                                            randomize=randomize,
                                            log=logs)

                # Initializing optimizer
                optimizer = SGDbidirectional(network=network, threshold=0.0001,
                                             init_learning_rate=learning_rate,
                                             tau=max_epochs * 2,
                                             final_learning_rate=learning_rate / 5.,
                                             init_learning_rate_backward=backward_learning_rate,
                                             final_learning_rate_backward=backward_learning_rate / 5.,
                                             compute_accuracies=False,
                                             max_epoch=max_epochs,
                                             outputfile_name='resultfile.csv')
                # Train on dataset

                try:
                    train_loss, test_loss = optimizer.run_dataset(input_dataset,
                                                                  output_dataset,
                                                                  input_dataset_test,
                                                                  output_dataset_test)
                    train_loss = hf.append_results(train_loss, max_epochs + 1)
                    test_loss = hf.append_results(test_loss, max_epochs + 1)
                    results_train[i, j, k, l, :] = train_loss
                    results_test[i, j, k, l, :] = test_loss
                    best_results[i, j, k, l] = np.min(test_loss)
                except Exception as e:
                    print('Training failed')
                    print('Occurred error:')
                    print(e)
                    succesful_run[i,j,k] = False
            #
np.save(log_dir + '/train_losses.npy', results_train)
np.save(log_dir + '/test_losses.npy', results_test)
np.save(log_dir + '/best_results.npy', best_results)
np.save(log_dir + '/succesful_run.npy', succesful_run)