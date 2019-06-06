"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD, SGDInvertible, SGDbidirectional
from layers.invertible_layer import InvertibleLinearOutputLayer, \
    InvertibleLeakyReluLayer, InvertibleInputLayer
from networks.invertible_network import InvertibleNetwork
from layers.modified_TP_layer import MTPInvertibleLinearOutputLayer, \
    MTPInvertibleLeakyReluLayer, MTPInvertibleInputLayer
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
import traceback

# good results for seed 32, no good results for seed 47
seed = 32
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
n = 6
n_out = 6
distance = 8.
CPU = True
debug = False
weight_decay = 0.0000
learning_rate_TP = 0.01
learning_rate_TPrandom = 0.01
learning_rate_MTP = 0.0005
learning_rate_BP = 0.001
learning_rate_fixed = 0.001
output_step_size = 0.1
randomize = True
max_epoch = 60
logs=True
threshold = 1e-10

random_iterations = 10
random_iteration = 0
# ======== set log directory ==========
log_dir_main = '../logs/final_combined_toy_example_unequal/'

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

# ======== Create result arrays ==========
train_losses_TPrandom = np.empty((random_iterations, max_epoch+1))
test_losses_TPrandom = np.empty((random_iterations, max_epoch+1))
approx_error_angles_array_TPrandom = np.empty((random_iterations, max_epoch*nb_training_batches))
approx_errors_array_TPrandom = np.empty((random_iterations, max_epoch*nb_training_batches))

train_losses_TP = np.empty((random_iterations, max_epoch+1))
test_losses_TP = np.empty((random_iterations, max_epoch+1))
approx_error_angles_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))
approx_errors_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))

train_losses_MTP = np.empty((random_iterations, max_epoch+1))
test_losses_MTP = np.empty((random_iterations, max_epoch+1))
approx_error_angles_array_MTP = np.empty((random_iterations, max_epoch*nb_training_batches))
approx_errors_array_MTP = np.empty((random_iterations, max_epoch*nb_training_batches))

train_losses_BP = np.empty((random_iterations, max_epoch+1))
test_losses_BP = np.empty((random_iterations, max_epoch+1))

train_losses_BP_fixed = np.empty((random_iterations, max_epoch+1))
test_losses_BP_fixed = np.empty((random_iterations, max_epoch+1))

while random_iteration < random_iterations:
    try:
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
        output_layer_true = LinearOutputLayer(in_dim=n, layer_dim=n_out,
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
            testing_size,1)

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


        # ===== Run experiment with randomized TP =======
        # initialize forward weights in the neighbourhood of true weights
        log_dir = log_dir_main + 'TP'
        writer = SummaryWriter(log_dir=log_dir)


        # Creating training network
        inputlayer_TPrandom = InvertibleInputLayer(layer_dim=n, out_dim=n, loss_function='mse',
                                          name='input_layer', writer=writer,
                                          debug_mode=debug,
                                          weight_decay=weight_decay)
        hiddenlayer_TPrandom = InvertibleLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                               layer_dim=n, out_dim=n_out, loss_function=
                                               'mse',
                                               name='hidden_layer',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay)
        outputlayer_TPrandom = InvertibleLinearOutputLayer(in_dim=n, layer_dim=n_out,
                                                  step_size=output_step_size,
                                                  name='output_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        hiddenlayer_TPrandom.set_forward_parameters(hidden_weights, hiddenlayer_TPrandom.forward_bias)
        outputlayer_TPrandom.set_forward_parameters(output_weights, outputlayer_TPrandom.forward_bias)

        network_TPrandom = InvertibleNetwork([inputlayer_TPrandom, hiddenlayer_TPrandom, outputlayer_TPrandom],
                                    randomize=randomize, log=logs)

        # Initializing optimizer
        optimizer1 = SGD(network=network_TPrandom, threshold=threshold,
                         init_learning_rate=learning_rate_TPrandom,
                         tau=max_epoch,
                         final_learning_rate=learning_rate_TPrandom/5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')

        # Train on dataset
        start_time = time.time()
        train_loss_TPrandom, test_loss_TPrandom = optimizer1.run_dataset(input_dataset, output_dataset, input_dataset_test,
                               output_dataset_test)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        approx_errors_TPrandom = hiddenlayer_TPrandom.approx_errors.numpy()
        approx_error_angle_TPrandom = hiddenlayer_TPrandom.approx_error_angles.numpy()

        train_losses_TPrandom[random_iteration,:] = train_loss_TPrandom
        test_losses_TPrandom[random_iteration,:] = test_loss_TPrandom
        approx_errors_array_TPrandom[random_iteration,:] = approx_errors_TPrandom
        approx_error_angles_array_TPrandom[random_iteration,:] = approx_error_angle_TPrandom

        # ===== Run experiment with TP =======
        # initialize forward weights in the neighbourhood of true weights
        log_dir = log_dir_main + 'TP'
        writer = SummaryWriter(log_dir=log_dir)

        # Creating training network
        inputlayer_TP = InvertibleInputLayer(layer_dim=n, out_dim=n,
                                             loss_function='mse',
                                             name='input_layer', writer=writer,
                                             debug_mode=debug,
                                             weight_decay=weight_decay)
        hiddenlayer_TP = InvertibleLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                                  layer_dim=n, out_dim=n_out,
                                                  loss_function=
                                                  'mse',
                                                  name='hidden_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        outputlayer_TP = InvertibleLinearOutputLayer(in_dim=n, layer_dim=n_out,
                                                     step_size=output_step_size,
                                                     name='output_layer',
                                                     writer=writer,
                                                     debug_mode=debug,
                                                     weight_decay=weight_decay)
        hiddenlayer_TP.set_forward_parameters(hidden_weights,
                                              hiddenlayer_TP.forward_bias)
        outputlayer_TP.set_forward_parameters(output_weights,
                                              outputlayer_TP.forward_bias)

        network_TP = InvertibleNetwork(
            [inputlayer_TP, hiddenlayer_TP, outputlayer_TP],
            randomize=randomize, log=logs)

        # Initializing optimizer
        optimizer1 = SGD(network=network_TP, threshold=threshold,
                         init_learning_rate=learning_rate_TP,
                         tau=max_epoch,
                         final_learning_rate=learning_rate_TP / 5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')

        # Train on dataset
        start_time = time.time()
        train_loss_TP, test_loss_TP = optimizer1.run_dataset(input_dataset,
                                                             output_dataset,
                                                             input_dataset_test,
                                                             output_dataset_test)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        approx_errors_TP = hiddenlayer_TP.approx_errors.numpy()
        approx_error_angle_TP = hiddenlayer_TP.approx_error_angles.numpy()

        train_losses_TP[random_iteration, :] = train_loss_TP
        test_losses_TP[random_iteration, :] = test_loss_TP
        approx_errors_array_TP[random_iteration, :] = approx_errors_TP
        approx_error_angles_array_TP[random_iteration,
        :] = approx_error_angle_TP

        # ====== MTP =====

        # Creating training network
        inputlayer_MTP = MTPInvertibleInputLayer(layer_dim=n, out_dim=n,
                                             loss_function='mse',
                                             name='input_layer', writer=writer,
                                             debug_mode=debug,
                                             weight_decay=weight_decay)
        hiddenlayer_MTP = MTPInvertibleLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                                  layer_dim=n, out_dim=n_out,
                                                  loss_function=
                                                  'mse',
                                                  name='hidden_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        outputlayer_MTP = MTPInvertibleLinearOutputLayer(in_dim=n, layer_dim=n_out,
                                                     step_size=output_step_size,
                                                     name='output_layer',
                                                     writer=writer,
                                                     debug_mode=debug,
                                                     weight_decay=weight_decay)
        hiddenlayer_MTP.set_forward_parameters(hidden_weights,
                                              hiddenlayer_MTP.forward_bias)
        outputlayer_MTP.set_forward_parameters(output_weights,
                                              outputlayer_MTP.forward_bias)

        network_MTP = InvertibleNetwork(
            [inputlayer_MTP, hiddenlayer_MTP, outputlayer_MTP],
            randomize=randomize, log=logs)

        # Initializing optimizer
        optimizer1 = SGD(network=network_MTP, threshold=threshold,
                         init_learning_rate=learning_rate_MTP,
                         tau=max_epoch,
                         final_learning_rate=learning_rate_MTP / 5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')

        # Train on dataset
        start_time = time.time()
        train_loss_MTP, test_loss_MTP = optimizer1.run_dataset(input_dataset,
                                                             output_dataset,
                                                             input_dataset_test,
                                                             output_dataset_test)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        approx_errors_MTP = hiddenlayer_MTP.approx_errors.numpy()
        approx_error_angle_MTP = hiddenlayer_MTP.approx_error_angles.numpy()

        train_losses_MTP[random_iteration, :] = train_loss_MTP
        test_losses_MTP[random_iteration, :] = test_loss_MTP
        approx_errors_array_MTP[random_iteration, :] = approx_errors_MTP
        approx_error_angles_array_MTP[random_iteration,
        :] = approx_error_angle_MTP

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
                                           weight_decay=weight_decay)
        outputlayer_BP = LinearOutputLayer(in_dim=n, layer_dim=n_out,
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

        train_losses_BP[random_iteration,:] = train_loss_BP
        test_losses_BP[random_iteration, :] = test_loss_BP

        # ===== Run experiment with fixed BP =======
        # initialize forward weights in the neighbourhood of true weights
        log_dir = log_dir_main + 'fixed_BP'
        writer = SummaryWriter(log_dir=log_dir)


        # Creating training network
        inputlayer_fixed_BP = InputLayer(layer_dim=n, writer=writer,
                                  name='input_layer',
                                  debug_mode=debug,
                                  weight_decay=weight_decay)
        hiddenlayer_fixed_BP = LeakyReluLayer(negative_slope=0.35, in_dim=n,
                                           layer_dim=n,
                                           writer=writer,
                                           name='hidden_layer',
                                           debug_mode=debug,
                                           weight_decay=weight_decay,
                                           fixed=True)
        outputlayer_fixed_BP = LinearOutputLayer(in_dim=n, layer_dim=n_out,
                                              loss_function='mse',
                                              writer=writer,
                                              name='output_layer',
                                              debug_mode=debug,
                                              weight_decay=weight_decay)
        hiddenlayer_fixed_BP.set_forward_parameters(hidden_weights, hiddenlayer_fixed_BP.forward_bias)
        outputlayer_fixed_BP.set_forward_parameters(output_weights, outputlayer_fixed_BP.forward_bias)

        network_fixed_BP = Network([inputlayer_fixed_BP,
                                     hiddenlayer_fixed_BP, outputlayer_fixed_BP], log=logs)

        # Initializing optimizer
        optimizer1 = SGD(network=network_fixed_BP, threshold=threshold,
                         init_learning_rate=learning_rate_fixed,
                         tau=100,
                         final_learning_rate=learning_rate_fixed/5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')

        # Train on dataset
        start_time = time.time()
        train_loss_fixed_BP, test_loss_fixed_BP = optimizer1.run_dataset(
            input_dataset, output_dataset, input_dataset_test,
                               output_dataset_test)
        end_time = time.time()
        train_losses_BP_fixed[random_iteration, :] = train_loss_fixed_BP
        test_losses_BP_fixed[random_iteration, :] = test_loss_fixed_BP
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        random_iteration += 1
    except Exception as e:
        print('Training failed')
        print('Occurred error:')
        print(e)


# ====== Save results =======
np.save(log_dir_main + 'train_losses_TP.npy', train_losses_TP)
np.save(log_dir_main + 'test_losses_TP.npy', test_losses_TP)
np.save(log_dir_main + 'approx_error_angles_array_TP.npy', approx_error_angles_array_TP)
np.save(log_dir_main + 'approx_errors_array_TP.npy', approx_errors_array_TP)

np.save(log_dir_main + 'train_losses_TPrandom.npy', train_losses_TPrandom)
np.save(log_dir_main + 'test_losses_TPrandom.npy', test_losses_TPrandom)
np.save(log_dir_main + 'approx_error_angles_array_TPrandom.npy', approx_error_angles_array_TPrandom)
np.save(log_dir_main + 'approx_errors_array_TPrandom.npy', approx_errors_array_TPrandom)

np.save(log_dir_main + 'train_losses_MTP.npy', train_losses_MTP)
np.save(log_dir_main + 'test_losses_MTP.npy', test_losses_MTP)
np.save(log_dir_main + 'approx_error_angles_array_MTP.npy', approx_error_angles_array_MTP)
np.save(log_dir_main + 'approx_errors_array_MTP.npy', approx_errors_array_MTP)

np.save(log_dir_main + 'train_losses_BP.npy', train_losses_BP)
np.save(log_dir_main + 'test_losses_BP.npy', test_losses_BP)

np.save(log_dir_main + 'train_losses_BP_fixed.npy', train_losses_BP_fixed)
np.save(log_dir_main + 'test_losses_BP_fixed.npy', test_losses_BP_fixed)

# ========= Average results ==========
train_loss_TP_mean = np.mean(train_losses_TP, axis=0)
test_loss_TP_mean = np.mean(test_losses_TP, axis=0)
approx_errors_TP_mean = np.mean(approx_errors_array_TP, axis=0)
approx_error_angle_TP_mean = np.mean(approx_error_angles_array_TP, axis=0)

train_loss_TPrandom_mean = np.mean(train_losses_TPrandom, axis=0)
test_loss_TPrandom_mean = np.mean(test_losses_TPrandom, axis=0)
approx_errors_TPrandom_mean = np.mean(approx_errors_array_TPrandom, axis=0)
approx_error_angle_TPrandom_mean = np.mean(approx_error_angles_array_TPrandom, axis=0)

train_loss_MTP_mean = np.mean(train_losses_MTP, axis=0)
test_loss_MTP_mean = np.mean(test_losses_MTP, axis=0)
approx_errors_MTP_mean = np.mean(approx_errors_array_MTP, axis=0)
approx_error_angle_MTP_mean = np.mean(approx_error_angles_array_MTP, axis=0)

train_loss_BP_mean = np.mean(train_losses_BP, axis=0)
test_loss_BP_mean = np.mean(test_losses_BP, axis=0)

train_loss_fixed_BP_mean = np.mean(train_losses_BP_fixed, axis=0)
test_loss_fixed_BP_mean = np.mean(test_losses_BP_fixed, axis=0)





# ========= PLOTS ===========
fontsize = 12
epochs = np.arange(0, max_epoch+1)
legend1 = ['invertible TP', 'invertible randomized TP', 'invertible MTP', 'BP', 'fixed layer BP']
legend2 = ['invertible TP', 'invertible randomized TP', 'invertible MTP']
# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure('training_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, train_loss_TP_mean)
plt.semilogy(epochs, train_loss_TPrandom_mean)
plt.semilogy(epochs, train_loss_MTP_mean)
plt.semilogy(epochs, train_loss_BP_mean)
plt.semilogy(epochs, train_loss_fixed_BP_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1)
plt.show()

fig = plt.figure('test_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, test_loss_TP_mean)
plt.semilogy(epochs, test_loss_TPrandom_mean)
plt.semilogy(epochs, test_loss_MTP_mean)
plt.semilogy(epochs, test_loss_BP_mean)
plt.semilogy(epochs, test_loss_fixed_BP_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1, fontsize=fontsize)
plt.show()

fig = plt.figure('approx_error_angles')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(approx_error_angle_TP_mean)
plt.plot(approx_error_angle_TPrandom_mean)
plt.plot(approx_error_angle_MTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()

fig = plt.figure('approx_errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(approx_errors_TP_mean)
plt.semilogy(approx_errors_TPrandom_mean)
plt.semilogy(approx_errors_MTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{approx}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize, loc='upper right')
plt.show()






