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
import traceback
import matplotlib.pyplot as plt

seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ======== User variables ============
nb_training_batches = 2
batch_size = 1
testing_size = 10
n = 6
distance = 8.
# learning_rates = [0.005, 0.001]
learning_rate = 0.0001
backward_learning_rate = 0.005
backward_weight_decay = 0.

output_step_size = 0.1
CPU = True
debug = False
weight_decay = 0.

max_epoch = 4
logs = False
threshold = 0.00001

random_iterations = 1
random_iteration = 0

# ======== set log directory ==========
log_dir = '../logs/toy_example_normal_DTP_4layers/'
writer = SummaryWriter(log_dir=log_dir)

# ======== Create result files ========
train_losses_true = np.empty((random_iterations, max_epoch+1))
test_losses_true = np.empty((random_iterations, max_epoch+1))
train_losses_false = np.empty((random_iterations, max_epoch+1))
test_losses_false = np.empty((random_iterations, max_epoch+1))



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
while random_iteration < random_iterations:
    try:

        input_layer_true = InputLayer(layer_dim=n, writer=writer,
                                      name='input_layer_true_model',
                                      debug_mode=debug,
                                      weight_decay=weight_decay)
        hidden_layer_true = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                           writer=writer,
                                           name='hidden_layer_true_model',
                                           debug_mode=debug,
                                           weight_decay=weight_decay)
        hidden_layer2_true = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                           writer=writer,
                                           name='hidden_layer2_true_model',
                                           debug_mode=debug,
                                           weight_decay=weight_decay)
        hidden_layer3_true = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                           writer=writer,
                                           name='hidden_layer3_true_model',
                                           debug_mode=debug,
                                           weight_decay=weight_decay)

        hidden_layer4_true = LeakyReluLayer(negative_slope=0.35, in_dim=n, layer_dim=n,
                                           writer=writer,
                                           name='hidden_layer4_true_model',
                                           debug_mode=debug,
                                           weight_decay=weight_decay)

        output_layer_true = LinearOutputLayer(in_dim=n, layer_dim=n,
                                              loss_function='mse',
                                              writer=writer,
                                              name='output_layer_true_model',
                                              debug_mode=debug,
                                              weight_decay=weight_decay)
        true_network = Network([input_layer_true, hidden_layer_true,
                                hidden_layer2_true,
                                hidden_layer3_true,
                                hidden_layer4_true,
                                output_layer_true])

        generator = GenerateDatasetFromModel(true_network)

        input_dataset, output_dataset = generator.generate(nb_training_batches,
                                                           batch_size)
        input_dataset_test, output_dataset_test = generator.generate(
            testing_size, 1)

        output_weights_true = output_layer_true.forward_weights
        hidden_weights_true = hidden_layer_true.forward_weights
        hidden_weights2_true = hidden_layer2_true.forward_weights
        hidden_weights3_true = hidden_layer3_true.forward_weights
        hidden_weights4_true = hidden_layer4_true.forward_weights

        output_weights = hf.get_invertible_neighbourhood_matrix(
            output_weights_true,
            distance)
        hidden_weights = hf.get_invertible_neighbourhood_matrix(
            hidden_weights_true,
            distance)
        hidden_weights2 = hf.get_invertible_neighbourhood_matrix(
            hidden_weights2_true,
            distance)
        hidden_weights3 = hf.get_invertible_neighbourhood_matrix(
            hidden_weights3_true,
            distance)
        hidden_weights4 = hf.get_invertible_neighbourhood_matrix(
            hidden_weights4_true,
            distance)



        inputlayer = InvertibleInputLayer(layer_dim=n, out_dim=n,
                                          loss_function='mse',
                                          name='input_layer', writer=writer,
                                          debug_mode=debug,
                                          weight_decay=weight_decay)
        hiddenlayer = InvertibleLeakyReluLayer(negative_slope=0.35,
                                               in_dim=n,
                                               layer_dim=n, out_dim=n,
                                               loss_function=
                                               'mse',
                                               name='hidden_layer',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay)
        hiddenlayer2 = InvertibleLeakyReluLayer(negative_slope=0.35,
                                               in_dim=n,
                                               layer_dim=n, out_dim=n,
                                               loss_function=
                                               'mse',
                                               name='hidden_layer2',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay)

        hiddenlayer3 = InvertibleLeakyReluLayer(negative_slope=0.35,
                                                in_dim=n,
                                                layer_dim=n, out_dim=n,
                                                loss_function=
                                                'mse',
                                                name='hidden_layer3',
                                                writer=writer,
                                                debug_mode=debug,
                                                weight_decay=weight_decay)

        hiddenlayer4 = InvertibleLeakyReluLayer(negative_slope=0.35,
                                                in_dim=n,
                                                layer_dim=n, out_dim=n,
                                                loss_function=
                                                'mse',
                                                name='hidden_layer4',
                                                writer=writer,
                                                debug_mode=debug,
                                                weight_decay=weight_decay)

        outputlayer = InvertibleLinearOutputLayer(in_dim=n, layer_dim=n,
                                                  step_size=output_step_size,
                                                  name='output_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        hiddenlayer.set_forward_parameters(hidden_weights,
                                           hiddenlayer.forward_bias)
        outputlayer.set_forward_parameters(output_weights,
                                           outputlayer.forward_bias)
        hiddenlayer2.set_forward_parameters(hidden_weights2,
                                            hiddenlayer2.forward_bias)
        hiddenlayer3.set_forward_parameters(hidden_weights3,
                                            hiddenlayer3.forward_bias)
        hiddenlayer4.set_forward_parameters(hidden_weights4,
                                            hiddenlayer4.forward_bias)

        network = InvertibleNetwork([inputlayer, hiddenlayer,
                                     hiddenlayer2,
                                     hiddenlayer3,
                                     hiddenlayer4,
                                     outputlayer],
                                    randomize=True,
                                    log=logs)

        # Initializing optimizer
        optimizer = SGD(network=network, threshold=threshold,
                                      init_learning_rate=learning_rate,
                                      tau=max_epoch,
                                      final_learning_rate=learning_rate / 5.,
                                      compute_accuracies=False,
                                      max_epoch=max_epoch,
                                      outputfile_name='resultfile.csv')
        # Train on dataset


        train_loss, test_loss = optimizer.run_dataset(input_dataset,
                                                      output_dataset,
                                                   input_dataset_test,
                                                   output_dataset_test)
        np.save(log_dir+'train_loss_true.npy', train_loss)
        np.save(log_dir+'test_loss_true.npy', test_loss)
        train_losses_true[random_iteration,:] = hf.append_results(train_loss, max_epoch+1)
        test_losses_true[random_iteration, :] = hf.append_results(test_loss, max_epoch+1)

        # Repeat for randomize=False

        inputlayer_false = InvertibleInputLayer(layer_dim=n, out_dim=n,
                                          loss_function='mse',
                                          name='input_layer', writer=writer,
                                          debug_mode=debug,
                                          weight_decay=weight_decay)
        hiddenlayer_false = InvertibleLeakyReluLayer(negative_slope=0.35,
                                               in_dim=n,
                                               layer_dim=n, out_dim=n,
                                               loss_function=
                                               'mse',
                                               name='hidden_layer',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay)
        hiddenlayer2_false = InvertibleLeakyReluLayer(negative_slope=0.35,
                                                in_dim=n,
                                                layer_dim=n, out_dim=n,
                                                loss_function=
                                                'mse',
                                                name='hidden_layer2',
                                                writer=writer,
                                                debug_mode=debug,
                                                weight_decay=weight_decay)

        hiddenlayer3_false = InvertibleLeakyReluLayer(negative_slope=0.35,
                                                in_dim=n,
                                                layer_dim=n, out_dim=n,
                                                loss_function=
                                                'mse',
                                                name='hidden_layer3',
                                                writer=writer,
                                                debug_mode=debug,
                                                weight_decay=weight_decay)

        hiddenlayer4_false = InvertibleLeakyReluLayer(negative_slope=0.35,
                                                in_dim=n,
                                                layer_dim=n, out_dim=n,
                                                loss_function=
                                                'mse',
                                                name='hidden_layer4',
                                                writer=writer,
                                                debug_mode=debug,
                                                weight_decay=weight_decay)

        outputlayer_false = InvertibleLinearOutputLayer(in_dim=n, layer_dim=n,
                                                  step_size=output_step_size,
                                                  name='output_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        hiddenlayer_false.set_forward_parameters(hidden_weights,
                                           hiddenlayer_false.forward_bias)
        outputlayer_false.set_forward_parameters(output_weights,
                                           outputlayer_false.forward_bias)
        hiddenlayer2_false.set_forward_parameters(hidden_weights2,
                                            hiddenlayer2_false.forward_bias)
        hiddenlayer3_false.set_forward_parameters(hidden_weights3,
                                            hiddenlayer3_false.forward_bias)
        hiddenlayer4_false.set_forward_parameters(hidden_weights4,
                                            hiddenlayer4_false.forward_bias)

        network_false = InvertibleNetwork([inputlayer_false, hiddenlayer_false,
                                     hiddenlayer2_false,
                                     hiddenlayer3_false,
                                     hiddenlayer4_false,
                                     outputlayer_false],
                                    randomize=False,
                                    log=logs)

        # Initializing optimizer
        optimizer = SGD(network=network_false, threshold=threshold,
                                     init_learning_rate=learning_rate,
                                     tau=max_epoch,
                                     final_learning_rate=learning_rate / 5.,
                                     compute_accuracies=False,
                                     max_epoch=max_epoch,
                                     outputfile_name='resultfile.csv')
        # Train on dataset

        train_loss, test_loss = optimizer.run_dataset(input_dataset,
                                                      output_dataset,
                                                      input_dataset_test,
                                                      output_dataset_test)
        train_losses_false[random_iteration, :] = hf.append_results(train_loss,
                                                                   max_epoch + 1)
        test_losses_false[random_iteration, :] = hf.append_results(test_loss,
                                                                  max_epoch + 1)

        random_iteration += 1
    except Exception as e:
        print('Training failed')
        print('Occurred error:')
        print(e)

# ===== Save results =======
np.save(log_dir + 'train_losses_true.npy', train_losses_true)
np.save(log_dir + 'test_losses_true.npy', test_losses_true)
np.save(log_dir + 'train_losses_true.npy', train_losses_true)
np.save(log_dir + 'test_losses_true.npy', test_losses_true)

# ====== Average results =======
train_loss_true_mean = np.mean(train_losses_true, axis=0)
train_loss_false_mean = np.mean(train_losses_false, axis=0)
test_loss_true_mean = np.mean(test_losses_true, axis=0)
test_loss_false_mean = np.mean(test_losses_false, axis=0)

# ==== Plots ======
fontsize = 12
epochs = np.arange(0, max_epoch+1)
legend1 = ['random layer updates', 'full layer updates']
legend2 = ['TP', 'DTP', 'original TP', 'original DTP']
# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure('training_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, train_loss_true_mean)
plt.semilogy(epochs, train_loss_false_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1)
plt.show()

fig = plt.figure('test_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, test_loss_true_mean)
plt.semilogy(epochs, test_loss_false_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1)
plt.show()
