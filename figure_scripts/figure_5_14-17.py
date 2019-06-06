"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD, SGDInvertible, SGDbidirectional
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
import traceback

# good results for seed 32, no good results for seed 47
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

"""
For figure 5.15, put randomize=False
For all other figures, put randomize=True
"""

# ======== User variables ============
nb_training_batches = 60
batch_size = 32
testing_size = 1000
n = 6
n_out = 6
distance = 8.
CPU = True
debug = False
weight_decay = 0.0000
learning_rate_TP = 0.09
learning_rate_BP = 0.01
learning_rate_DTP = 0.1
learning_rate_fixed = 0.01
learning_rate_originalTP = 0.01
learning_rate_originalDTP = 0.1
weight_decay_TP = 0.0
weight_decay_DTP = 0.0
weight_decay_originalTP = 0.0
weight_decay_originalDTP = 0.0
backward_learning_rate_TP = 0.08
backward_learning_rate_DTP = 0.05
backward_learning_rate_originalTP = 0.01
backward_learning_rate_originalDTP = 0.01

output_step_size = 0.1
randomize = True
max_epoch = 60
logs=True
threshold = 0.0000001

random_iterations = 15
random_iteration = 0
# ======== set log directory ==========
log_dir_main = '../logs/final_combined_toy_example_randomize_false/'

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
train_losses_TP = np.empty((random_iterations, max_epoch+1))
test_losses_TP = np.empty((random_iterations, max_epoch+1))
approx_error_angles_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))
approx_errors_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_errors_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))
TP_errors_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_angles_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))
BP_angles_array_TP = np.empty((random_iterations, max_epoch*nb_training_batches))

train_losses_DTP = np.empty((random_iterations, max_epoch+1))
test_losses_DTP = np.empty((random_iterations, max_epoch+1))
approx_error_angles_array_DTP = np.empty((random_iterations, max_epoch*nb_training_batches))
approx_errors_array_DTP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_errors_array_DTP = np.empty((random_iterations, max_epoch*nb_training_batches))
TP_errors_array_DTP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_angles_array_DTP = np.empty((random_iterations, max_epoch*nb_training_batches))
BP_angles_array_DTP = np.empty((random_iterations, max_epoch*nb_training_batches))

train_losses_originalTP = np.empty((random_iterations, max_epoch+1))
test_losses_originalTP = np.empty((random_iterations, max_epoch+1))
approx_error_angles_array_originalTP = np.empty((random_iterations, max_epoch*nb_training_batches))
approx_errors_array_originalTP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_errors_array_originalTP = np.empty((random_iterations, max_epoch*nb_training_batches))
TP_errors_array_originalTP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_angles_array_originalTP = np.empty((random_iterations, max_epoch*nb_training_batches))
BP_angles_array_originalTP = np.empty((random_iterations, max_epoch*nb_training_batches))

train_losses_originalDTP = np.empty((random_iterations, max_epoch+1))
test_losses_originalDTP = np.empty((random_iterations, max_epoch+1))
approx_error_angles_array_originalDTP = np.empty((random_iterations, max_epoch*nb_training_batches))
approx_errors_array_originalDTP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_errors_array_originalDTP = np.empty((random_iterations, max_epoch*nb_training_batches))
TP_errors_array_originalDTP = np.empty((random_iterations, max_epoch*nb_training_batches))
GN_angles_array_originalDTP = np.empty((random_iterations, max_epoch*nb_training_batches))
BP_angles_array_originalDTP = np.empty((random_iterations, max_epoch*nb_training_batches))

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
            1, testing_size)

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


        # ===== Run experiment with TP =======
        # initialize forward weights in the neighbourhood of true weights
        log_dir = log_dir_main + 'TP'
        writer = SummaryWriter(log_dir=log_dir)


        # Creating training network
        inputlayer_TP = TargetPropInputLayer(layer_dim=n, out_dim=n, loss_function='mse',
                                          name='input_layer', writer=writer,
                                          debug_mode=debug,
                                          weight_decay=weight_decay,
                                             weight_decay_backward=weight_decay_TP)
        hiddenlayer_TP = TargetPropLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                               layer_dim=n, out_dim=n_out, loss_function=
                                               'mse',
                                               name='hidden_layer',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay,
                                                  weight_decay_backward=weight_decay_TP)
        outputlayer_TP = TargetPropLinearOutputLayer(in_dim=n, layer_dim=n_out,
                                                  step_size=output_step_size,
                                                  name='output_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        hiddenlayer_TP.set_forward_parameters(hidden_weights, hiddenlayer_TP.forward_bias)
        outputlayer_TP.set_forward_parameters(output_weights, outputlayer_TP.forward_bias)

        network_TP = TargetPropNetwork([inputlayer_TP, hiddenlayer_TP, outputlayer_TP],
                                    randomize=randomize, log=logs)

        # Initializing optimizer
        optimizer1 = SGDbidirectional(network=network_TP, threshold=threshold,
                         init_learning_rate=learning_rate_TP,
                         tau=max_epoch,
                         final_learning_rate=learning_rate_TP/5.,
                         init_learning_rate_backward=backward_learning_rate_TP,
                         final_learning_rate_backward=backward_learning_rate_TP/5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')

        # Train on dataset
        start_time = time.time()
        train_loss_TP, test_loss_TP = optimizer1.run_dataset(input_dataset, output_dataset, input_dataset_test,
                               output_dataset_test)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        approx_errors_TP = hiddenlayer_TP.approx_errors.numpy()
        approx_error_angle_TP = hiddenlayer_TP.approx_error_angles.numpy()
        GN_errors_TP = hiddenlayer_TP.GN_errors.numpy()
        TP_errors_TP = hiddenlayer_TP.TP_errors.numpy()
        GN_angles_TP = hiddenlayer_TP.GN_angles.numpy()
        BP_angles_TP = hiddenlayer_TP.BP_angles.numpy()

        train_losses_TP[random_iteration,:] = train_loss_TP
        test_losses_TP[random_iteration,:] = test_loss_TP
        approx_errors_array_TP[random_iteration,:] = approx_errors_TP
        approx_error_angles_array_TP[random_iteration,:] = approx_error_angle_TP
        GN_errors_array_TP[random_iteration,:] = GN_errors_TP
        TP_errors_array_TP[random_iteration,:] = TP_errors_TP
        GN_angles_array_TP[random_iteration,:] = GN_angles_TP
        BP_angles_array_TP[random_iteration, :] = BP_angles_TP

        # ===== Run experiment with DTP =======
        # initialize forward weights in the neighbourhood of true weights
        log_dir = log_dir_main + 'DTP'
        writer = SummaryWriter(log_dir=log_dir)


        # Creating training network
        inputlayer_DTP =DTPInputLayer(layer_dim=n, out_dim=n, loss_function='mse',
                                          name='input_layer', writer=writer,
                                          debug_mode=debug,
                                          weight_decay=weight_decay,
                                             weight_decay_backward=weight_decay_DTP)
        hiddenlayer_DTP = DTPLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                               layer_dim=n, out_dim=n_out, loss_function=
                                               'mse',
                                               name='hidden_layer',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay,
                                                  weight_decay_backward=weight_decay_DTP)
        outputlayer_DTP = DTPLinearOutputLayer(in_dim=n, layer_dim=n_out,
                                                  step_size=output_step_size,
                                                  name='output_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        hiddenlayer_DTP.set_forward_parameters(hidden_weights, hiddenlayer_DTP.forward_bias)
        outputlayer_DTP.set_forward_parameters(output_weights, outputlayer_DTP.forward_bias)

        network_DTP = TargetPropNetwork([inputlayer_DTP, hiddenlayer_DTP, outputlayer_DTP],
                                    randomize=randomize, log=logs)

        # Initializing optimizer
        optimizer1 = SGDbidirectional(network=network_DTP, threshold=threshold,
                         init_learning_rate=learning_rate_DTP,
                         tau=max_epoch,
                         final_learning_rate=learning_rate_DTP/5.,
                         init_learning_rate_backward=backward_learning_rate_DTP,
                         final_learning_rate_backward=backward_learning_rate_DTP/5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')

        # Train on dataset
        start_time = time.time()
        train_loss_DTP, test_loss_DTP = optimizer1.run_dataset(input_dataset, output_dataset, input_dataset_test,
                               output_dataset_test)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        approx_errors_DTP = hiddenlayer_DTP.approx_errors.numpy()
        approx_error_angle_DTP = hiddenlayer_DTP.approx_error_angles.numpy()
        GN_errors_DTP = hiddenlayer_DTP.GN_errors.numpy()
        TP_errors_DTP = hiddenlayer_DTP.TP_errors.numpy()
        GN_angles_DTP = hiddenlayer_DTP.GN_angles.numpy()
        BP_angles_DTP = hiddenlayer_DTP.BP_angles.numpy()

        train_losses_DTP[random_iteration, :] = train_loss_DTP
        test_losses_DTP[random_iteration, :] = test_loss_DTP
        approx_errors_array_DTP[random_iteration, :] = approx_errors_DTP
        approx_error_angles_array_DTP[random_iteration, :] = approx_error_angle_DTP
        GN_errors_array_DTP[random_iteration, :] = GN_errors_DTP
        TP_errors_array_DTP[random_iteration, :] = TP_errors_DTP
        GN_angles_array_DTP[random_iteration, :] = GN_angles_DTP
        BP_angles_array_DTP[random_iteration, :] = BP_angles_DTP

        # ===== Run experiment with original TP =======
        # initialize forward weights in the neighbourhood of true weights
        log_dir = log_dir_main + 'original_TP'
        writer = SummaryWriter(log_dir=log_dir)


        # Creating training network
        inputlayer_OriginalTP = OriginalTPInputLayer(layer_dim=n, out_dim=n, loss_function='mse',
                                          name='input_layer', writer=writer,
                                          debug_mode=debug,
                                          weight_decay=weight_decay,
                                             weight_decay_backward=weight_decay_originalTP)
        hiddenlayer_OriginalTP = OriginalTPLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                               layer_dim=n, out_dim=n_out, loss_function=
                                               'mse',
                                               name='hidden_layer',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay,
                                                  weight_decay_backward=weight_decay_originalTP)
        outputlayer_OriginalTP = OriginalTPLinearOutputLayer(in_dim=n, layer_dim=n_out,
                                                  step_size=output_step_size,
                                                  name='output_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        hiddenlayer_OriginalTP.set_forward_parameters(hidden_weights, hiddenlayer_OriginalTP.forward_bias)
        outputlayer_OriginalTP.set_forward_parameters(output_weights, outputlayer_OriginalTP.forward_bias)

        network_originalTP = TargetPropNetwork([inputlayer_OriginalTP, hiddenlayer_OriginalTP, outputlayer_OriginalTP],
                                    randomize=randomize, log=logs)

        # Initializing optimizer
        optimizer1 = SGDbidirectional(network=network_originalTP, threshold=threshold,
                         init_learning_rate=learning_rate_originalTP,
                         tau=max_epoch,
                         final_learning_rate=learning_rate_originalTP/5.,
                         init_learning_rate_backward=backward_learning_rate_originalTP,
                         final_learning_rate_backward=backward_learning_rate_originalTP/5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')

        # Train on dataset
        start_time = time.time()
        train_loss_originalTP, test_loss_originalTP = optimizer1.run_dataset(
            input_dataset, output_dataset, input_dataset_test,
                               output_dataset_test)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        approx_errors_originalTP = hiddenlayer_OriginalTP.approx_errors.numpy()
        approx_error_angle_originalTP = hiddenlayer_OriginalTP.approx_error_angles.numpy()
        GN_errors_originalTP = hiddenlayer_OriginalTP.GN_errors.numpy()
        TP_errors_originalTP = hiddenlayer_OriginalTP.TP_errors.numpy()
        GN_angles_originalTP = hiddenlayer_OriginalTP.GN_angles.numpy()
        BP_angles_originalTP = hiddenlayer_OriginalTP.BP_angles.numpy()

        train_losses_originalTP[random_iteration, :] = train_loss_originalTP
        test_losses_originalTP[random_iteration, :] = test_loss_originalTP
        approx_errors_array_originalTP[random_iteration, :] = approx_errors_originalTP
        approx_error_angles_array_originalTP[random_iteration, :] = approx_error_angle_originalTP
        GN_errors_array_originalTP[random_iteration, :] = GN_errors_originalTP
        TP_errors_array_originalTP[random_iteration, :] = TP_errors_originalTP
        GN_angles_array_originalTP[random_iteration, :] = GN_angles_originalTP
        BP_angles_array_originalTP[random_iteration, :] = BP_angles_originalTP

        # ===== Run experiment with original DTP =======
        # initialize forward weights in the neighbourhood of true weights
        log_dir = log_dir_main + 'original_DTP'
        writer = SummaryWriter(log_dir=log_dir)


        # Creating training network
        inputlayer_OriginalDTP = OriginalDTPInputLayer(layer_dim=n, out_dim=n, loss_function='mse',
                                          name='input_layer', writer=writer,
                                          debug_mode=debug,
                                          weight_decay=weight_decay,
                                             weight_decay_backward=weight_decay_originalDTP)
        hiddenlayer_OriginalDTP = OriginalDTPLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                               layer_dim=n, out_dim=n_out, loss_function=
                                               'mse',
                                               name='hidden_layer',
                                               writer=writer,
                                               debug_mode=debug,
                                               weight_decay=weight_decay,
                                                  weight_decay_backward=weight_decay_originalDTP)
        outputlayer_OriginalDTP = OriginalDTPLinearOutputLayer(in_dim=n, layer_dim=n_out,
                                                  step_size=output_step_size,
                                                  name='output_layer',
                                                  writer=writer,
                                                  debug_mode=debug,
                                                  weight_decay=weight_decay)
        hiddenlayer_OriginalDTP.set_forward_parameters(hidden_weights, hiddenlayer_OriginalDTP.forward_bias)
        outputlayer_OriginalDTP.set_forward_parameters(output_weights, outputlayer_OriginalDTP.forward_bias)

        network_originalDTP = TargetPropNetwork([inputlayer_OriginalDTP,
                                     hiddenlayer_OriginalDTP, outputlayer_OriginalDTP],
                                    randomize=randomize, log=logs)

        # Initializing optimizer
        optimizer1 = SGDbidirectional(network=network_originalDTP, threshold=threshold,
                         init_learning_rate=learning_rate_originalDTP,
                         tau=max_epoch,
                         final_learning_rate=learning_rate_originalDTP/5.,
                         init_learning_rate_backward=backward_learning_rate_originalDTP,
                         final_learning_rate_backward=backward_learning_rate_originalDTP/5.,
                         compute_accuracies=False,
                         max_epoch=max_epoch,
                         outputfile_name='resultfile.csv')


        # Train on dataset
        start_time = time.time()
        train_loss_originalDTP, test_loss_originalDTP = optimizer1.run_dataset(
            input_dataset, output_dataset, input_dataset_test,
                               output_dataset_test)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        approx_errors_originalDTP = hiddenlayer_OriginalDTP.approx_errors.numpy()
        approx_error_angle_originalDTP = hiddenlayer_OriginalDTP.approx_error_angles.numpy()
        GN_errors_originalDTP = hiddenlayer_OriginalDTP.GN_errors.numpy()
        TP_errors_originalDTP = hiddenlayer_OriginalDTP.TP_errors.numpy()
        GN_angles_originalDTP = hiddenlayer_OriginalDTP.GN_angles.numpy()
        BP_angles_originalDTP = hiddenlayer_OriginalDTP.BP_angles.numpy()

        train_losses_originalDTP[random_iteration, :] = train_loss_originalDTP
        test_losses_originalDTP[random_iteration, :] = test_loss_originalDTP
        approx_errors_array_originalDTP[random_iteration,
        :] = approx_errors_originalDTP
        approx_error_angles_array_originalDTP[random_iteration,
        :] = approx_error_angle_originalDTP
        GN_errors_array_originalDTP[random_iteration, :] = GN_errors_originalDTP
        TP_errors_array_originalDTP[random_iteration, :] = TP_errors_originalDTP
        GN_angles_array_originalDTP[random_iteration, :] = GN_angles_originalDTP
        BP_angles_array_originalDTP[random_iteration, :] = BP_angles_originalDTP

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
np.save(log_dir_main + 'GN_errors_array_TP.npy', GN_errors_array_TP)
np.save(log_dir_main + 'TP_errors_array_TP.npy', TP_errors_array_TP)
np.save(log_dir_main + 'GN_angles_array_TP.npy', GN_angles_array_TP)
np.save(log_dir_main + 'BP_angles_array_TP.npy', BP_angles_array_TP)

np.save(log_dir_main + 'train_losses_DTP.npy', train_losses_DTP)
np.save(log_dir_main + 'test_losses_DTP.npy', test_losses_DTP)
np.save(log_dir_main + 'approx_error_angles_array_DTP.npy', approx_error_angles_array_DTP)
np.save(log_dir_main + 'approx_errors_array_DTP.npy', approx_errors_array_DTP)
np.save(log_dir_main + 'GN_errors_array_DTP.npy', GN_errors_array_DTP)
np.save(log_dir_main + 'TP_errors_array_DTP.npy', TP_errors_array_DTP)
np.save(log_dir_main + 'GN_angles_array_DTP.npy', GN_angles_array_DTP)
np.save(log_dir_main + 'BP_angles_array_DTP.npy', BP_angles_array_DTP)

np.save(log_dir_main + 'train_losses_originalTP.npy', train_losses_originalTP)
np.save(log_dir_main + 'test_losses_originalTP.npy', test_losses_originalTP)
np.save(log_dir_main + 'approx_error_angles_array_originalTP.npy', approx_error_angles_array_originalTP)
np.save(log_dir_main + 'approx_errors_array_originalTP.npy', approx_errors_array_originalTP)
np.save(log_dir_main + 'GN_errors_array_originalTP.npy', GN_errors_array_originalTP)
np.save(log_dir_main + 'TP_errors_array_originalTP.npy', TP_errors_array_originalTP)
np.save(log_dir_main + 'GN_angles_array_originalTP.npy', GN_angles_array_originalTP)
np.save(log_dir_main + 'BP_angles_array_originalTP.npy', BP_angles_array_originalTP)

np.save(log_dir_main + 'train_losses_originalDTP.npy', train_losses_originalDTP)
np.save(log_dir_main + 'test_losses_originalDTP.npy', test_losses_originalDTP)
np.save(log_dir_main + 'approx_error_angles_array_originalDTP.npy', approx_error_angles_array_originalDTP)
np.save(log_dir_main + 'approx_errors_array_originalDTP.npy', approx_errors_array_originalDTP)
np.save(log_dir_main + 'GN_errors_array_originalDTP.npy', GN_errors_array_originalDTP)
np.save(log_dir_main + 'TP_errors_array_originalDTP.npy', TP_errors_array_originalDTP)
np.save(log_dir_main + 'GN_angles_array_originalDTP.npy', GN_angles_array_originalDTP)
np.save(log_dir_main + 'BP_angles_array_originalDTP.npy', BP_angles_array_originalDTP)

np.save(log_dir_main + 'train_losses_BP.npy', train_losses_BP)
np.save(log_dir_main + 'test_losses_BP.npy', test_losses_BP)

np.save(log_dir_main + 'train_losses_BP_fixed.npy', train_losses_BP_fixed)
np.save(log_dir_main + 'test_losses_BP_fixed.npy', test_losses_BP_fixed)

# ========= Average results ==========
train_loss_TP_mean = np.mean(train_losses_TP, axis=0)
test_loss_TP_mean = np.mean(test_losses_TP, axis=0)
approx_errors_TP_mean = np.mean(approx_errors_array_TP, axis=0)
approx_error_angle_TP_mean = np.mean(approx_error_angles_array_TP, axis=0)
inverse_fraction_learning_signal_TP = np.mean(approx_errors_array_TP/GN_errors_array_TP, axis=0)
GN_errors_TP_mean = np.mean(GN_errors_array_TP, axis=0)
GN_angles_TP_mean = np.mean(GN_angles_array_TP, axis=0)
BP_angles_TP_mean = np.mean(BP_angles_array_TP, axis=0)

train_loss_DTP_mean = np.mean(train_losses_DTP, axis=0)
test_loss_DTP_mean = np.mean(test_losses_DTP, axis=0)
approx_errors_DTP_mean = np.mean(approx_errors_array_DTP, axis=0)
approx_error_angle_DTP_mean = np.mean(approx_error_angles_array_DTP, axis=0)
inverse_fraction_learning_signal_DTP = np.mean(approx_errors_array_DTP/GN_errors_array_DTP, axis=0)
GN_errors_DTP_mean = np.mean(GN_errors_array_DTP, axis=0)
GN_angles_DTP_mean = np.mean(GN_angles_array_DTP, axis=0)
BP_angles_DTP_mean = np.mean(BP_angles_array_DTP, axis=0)

train_loss_originalTP_mean = np.mean(train_losses_originalTP, axis=0)
test_loss_originalTP_mean = np.mean(test_losses_originalTP, axis=0)
approx_errors_originalTP_mean = np.mean(approx_errors_array_originalTP, axis=0)
approx_error_angle_originalTP_mean = np.mean(approx_error_angles_array_originalTP, axis=0)
inverse_fraction_learning_signal_originalTP = np.mean(approx_errors_array_originalTP/GN_errors_array_originalTP, axis=0)
GN_errors_originalTP_mean = np.mean(GN_errors_array_originalTP, axis=0)
GN_angles_originalTP_mean = np.mean(GN_angles_array_originalTP, axis=0)
BP_angles_originalTP_mean = np.mean(BP_angles_array_originalTP, axis=0)

train_loss_originalDTP_mean = np.mean(train_losses_originalDTP, axis=0)
test_loss_originalDTP_mean = np.mean(test_losses_originalDTP, axis=0)
approx_errors_originalDTP_mean = np.mean(approx_errors_array_originalDTP, axis=0)
approx_error_angle_originalDTP_mean = np.mean(approx_error_angles_array_originalDTP, axis=0)
inverse_fraction_learning_signal_originalDTP = np.mean(approx_errors_array_originalDTP/GN_errors_array_originalDTP, axis=0)
GN_errors_originalDTP_mean = np.mean(GN_errors_array_originalDTP, axis=0)
GN_angles_originalDTP_mean = np.mean(GN_angles_array_originalDTP, axis=0)
BP_angles_originalDTP_mean = np.mean(BP_angles_array_originalDTP, axis=0)

train_loss_BP_mean = np.mean(train_losses_BP, axis=0)
test_loss_BP_mean = np.mean(test_losses_BP, axis=0)

train_loss_fixed_BP_mean = np.mean(train_losses_BP_fixed, axis=0)
test_loss_fixed_BP_mean = np.mean(test_losses_BP_fixed, axis=0)





# ========= PLOTS ===========
fontsize = 12
epochs = np.arange(0, max_epoch+1)
legend1 = ['TP', 'DTP', 'original TP', 'original DTP', 'BP', 'fixed layer BP']
legend2 = ['TP', 'DTP', 'original TP', 'original DTP']
# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure('training_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, train_loss_TP_mean)
plt.semilogy(epochs, train_loss_DTP_mean)
plt.semilogy(epochs, train_loss_originalTP_mean)
plt.semilogy(epochs, train_loss_originalDTP_mean)
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
plt.semilogy(epochs, test_loss_DTP_mean)
plt.semilogy(epochs, test_loss_originalTP_mean)
plt.semilogy(epochs, test_loss_originalDTP_mean)
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
plt.plot(approx_error_angle_DTP_mean)
plt.plot(approx_error_angle_originalTP_mean)
plt.plot(approx_error_angle_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()

fig = plt.figure('approx_errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(approx_errors_TP_mean)
plt.semilogy(approx_errors_DTP_mean)
plt.semilogy(approx_errors_originalTP_mean)
plt.semilogy(approx_errors_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{approx}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize, loc='upper right')
plt.show()

fig = plt.figure('learning_signal')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(GN_errors_TP_mean)
plt.semilogy(GN_errors_DTP_mean)
plt.semilogy(GN_errors_originalTP_mean)
plt.semilogy(GN_errors_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{TP}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()

fig = plt.figure('GN angles')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(GN_angles_TP_mean)
plt.plot(GN_angles_DTP_mean)
plt.plot(GN_angles_originalTP_mean)
plt.plot(GN_angles_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()

fig = plt.figure('BP angles')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(BP_angles_TP_mean)
plt.plot(BP_angles_DTP_mean)
plt.plot(BP_angles_originalTP_mean)
plt.plot(BP_angles_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()


fig = plt.figure('learning_signal_ratio_inverse')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(inverse_fraction_learning_signal_TP)
plt.plot(inverse_fraction_learning_signal_DTP)
plt.plot(inverse_fraction_learning_signal_originalTP)
plt.plot(inverse_fraction_learning_signal_originalDTP)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{approx}\|_2/ \|e^{TP}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()





