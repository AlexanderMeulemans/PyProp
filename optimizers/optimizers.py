"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from utils import helper_functions as hf
from networks.network import Network
import pandas as pd
from networks.invertible_network import InvertibleNetwork


class Optimizer(object):
    """" Super class for all the different optimizers (e.g. SGD)"""

    def __init__(self, network, max_epoch=150, compute_accuracies=False,
                 outputfile_name='result_file.csv'):
        """
        :param network: network to train
        :param compute_accuracies: True if the optimizer should also save
        the accuracies. Only possible with
        classification problems
        :type network: Network
        """
        self.epoch = 0
        self.epoch_losses = torch.Tensor([])
        self.batch_losses = torch.Tensor([])
        self.single_batch_losses = torch.Tensor([])
        self.test_losses = torch.Tensor([])
        self.test_batch_losses = torch.Tensor([])
        self.set_network(network)
        self.set_compute_accuracies(compute_accuracies)
        self.set_max_epoch(max_epoch)
        self.writer = self.network.writer
        self.global_step = 0
        self.outputfile = pd.DataFrame(columns=['Train_loss', 'Test_loss'])
        self.outputfile_name = '../logs/{}'.format(outputfile_name)
        if self.compute_accuracies:
            self.epoch_accuracies = torch.Tensor([])
            self.batch_accuracies = torch.Tensor([])
            self.single_batch_accuracies = torch.Tensor([])
            self.test_accuracies = torch.Tensor([])
            self.test_batch_accuracies = torch.Tensor([])
            self.outputfile = pd.DataFrame(columns=
                                           ['Train_loss', 'Test_loss',
                                            'Train_accuracy', 'Test_accuracy'])

    def set_network(self, network):
        if not isinstance(network, Network):
            raise TypeError("Expecting Network object, instead got "
                            "{}".format(type(network)))
        self.network = network

    def set_learning_rate(self, learning_rate):
        if not isinstance(learning_rate, float):
            raise TypeError("Expecting a float number as learning_rate")
        if learning_rate <= 0:
            raise ValueError("Expecting a strictly positive learning rate")
        self.learning_rate = learning_rate

    def set_threshold(self, threshold):
        if not isinstance(threshold, float):
            raise TypeError("Expecting a float number as threshold")
        if threshold <= 0:
            raise ValueError("Expecting a strictly positive threshold")
        self.threshold = threshold

    def set_compute_accuracies(self, compute_accuracies):
        if not isinstance(compute_accuracies, bool):
            raise TypeError("Expecting a bool as compute_accuracies")
        self.compute_accuracies = compute_accuracies

    def set_max_epoch(self, max_epoch):
        if not isinstance(max_epoch, int):
            raise TypeError('Expecting integer for max_epoch, got '
                            '{}'.format(type(max_epoch)))
        if max_epoch <= 0:
            raise ValueError('Expecting strictly positive integer for '
                             'max_epoch, got {}'.format(max_epoch))
        self.max_epoch = max_epoch

    def reset_single_batch_losses(self):
        self.single_batch_losses = torch.Tensor([])

    def reset_single_batch_accuracies(self):
        self.single_batch_accuracies = torch.Tensor([])

    def reset_test_batch_losses(self):
        self.test_batch_losses = torch.Tensor([])

    def reset_test_batch_accuracies(self):
        self.test_batch_accuracies = torch.Tensor([])

    def reset_optimizer(self):
        self.epoch = 0
        self.epoch_losses = torch.Tensor([])
        self.batch_losses = torch.Tensor([])
        self.single_batch_losses = torch.Tensor([])
        self.test_losses = torch.Tensor([])
        self.test_batch_losses = torch.Tensor([])
        self.global_step = 0
        if self.compute_accuracies:
            self.epoch_accuracies = torch.Tensor([])
            self.batch_accuracies = torch.Tensor([])
            self.single_batch_accuracies = torch.Tensor([])
            self.test_accuracies = torch.Tensor([])
            self.test_batch_accuracies = torch.Tensor([])

    def update_learning_rate(self):
        """ If the optimizer should do a specific update of the learningrate,
        this method should be overwritten in the subclass"""
        pass

    def save_results(self, targets):
        """ Save the results of the optimizing step in the optimizer object."""
        loss = self.network.loss(targets)
        self.writer.add_scalar(tag='training_loss_batch',
                               scalar_value=loss,
                               global_step=self.global_step)
        self.batch_losses = torch.cat([self.batch_losses, loss])
        self.single_batch_losses = torch.cat([self.single_batch_losses, loss])
        self.network.save_state(self.global_step)
        if self.compute_accuracies:
            accuracy = self.network.accuracy(targets)
            self.writer.add_scalar(tag='training_accuracy_batch',
                                   scalar_value=accuracy,
                                   global_step=self.global_step)
            self.batch_accuracies = torch.cat([self.batch_accuracies, accuracy],
                                              0)
            self.single_batch_accuracies = torch.cat(
                [self.single_batch_accuracies, accuracy], 0)

    def test_step(self, data, target):
        self.network.propagate_forward(data)
        self.save_test_results_batch(target)

    def save_test_results_batch(self, target):
        batch_loss = self.network.loss(target)
        self.test_batch_losses = torch.cat([self.test_batch_losses,
                                            batch_loss], 0)
        if self.compute_accuracies:
            batch_accuracy = self.network.accuracy(target)
            self.test_batch_accuracies = torch.cat([self.test_batch_accuracies,
                                                    batch_accuracy])

    def save_test_results_epoch(self):
        test_loss = torch.Tensor([torch.mean(self.test_batch_losses)])
        self.test_losses = torch.cat([self.test_losses, test_loss], 0)
        self.writer.add_scalar(tag='test_loss',
                               scalar_value=test_loss,
                               global_step=self.epoch)
        self.reset_test_batch_losses()
        print('Test Loss: ' + str(test_loss))
        if self.compute_accuracies:
            test_accuracy = torch.Tensor([torch.mean(
                self.test_batch_accuracies)])
            self.test_accuracies = torch.cat(
                [self.test_accuracies, test_accuracy], 0)
            self.writer.add_scalar(tag='test_accuracy',
                                   scalar_value=test_accuracy,
                                   global_step=self.epoch)
            self.reset_test_batch_accuracies()
            print('Test Accuracy: ' + str(test_accuracy))

    def save_train_results_epoch(self):
        epoch_loss = torch.Tensor([torch.mean(self.single_batch_losses)])
        self.writer.add_scalar(tag='train_loss', scalar_value=epoch_loss,
                               global_step=self.epoch)
        self.reset_single_batch_losses()
        self.epoch_losses = torch.cat([self.epoch_losses, epoch_loss], 0)
        self.network.save_state_histograms(self.epoch)
        print('Train Loss: ' + str(epoch_loss))
        if self.compute_accuracies:
            epoch_accuracy = torch.Tensor([torch.mean(
                self.single_batch_accuracies)
                                          ])
            self.epoch_accuracies = torch.cat([self.epoch_accuracies,
                                               epoch_accuracy], 0)
            self.writer.add_scalar(tag='train_accuracy',
                                   scalar_value=epoch_accuracy,
                                   global_step=self.epoch)
            self.reset_single_batch_accuracies()
            print('Train Accuracy: ' + str(epoch_accuracy))

    def save_result_file(self):
        train_loss = self.epoch_losses[-1]
        test_loss = self.test_losses[-1]
        if self.compute_accuracies:
            train_accuracy = self.epoch_accuracies[-1]
            test_accuracy = self.test_accuracies[-1]
            self.outputfile.loc[self.epoch] = [train_loss, test_loss,
                                               train_accuracy, test_accuracy]
        else:
            self.outputfile.loc[self.epoch] = [train_loss, test_loss]

    def run_mnist(self, train_loader, test_loader, device):
        """ Train the network on the total training set of MNIST as
        long as epoch loss is above the threshold
        :param train_loader: a torch.utils.data.DataLoader object
        which containts the dataset"""
        if not isinstance(train_loader, torch.utils.data.DataLoader):
            raise TypeError("Expecting a DataLoader object, now got a "
                            "{}".format(type(train_loader)))

        epoch_loss = float('inf')
        print('====== Training started =======')
        print('Epoch: ' + str(self.epoch) + ' ------------------------')
        while epoch_loss > self.threshold and self.epoch < self.max_epoch:
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx % 200 == 0:
                    print('batch: ' + str(batch_idx))
                data = data.view(-1, 28 * 28, 1)
                target = hf.one_hot(target, 10)
                data, target = data.to(device), target.to(device)
                self.step(data, target)
            self.save_train_results_epoch()
            epoch_loss = self.epoch_losses[-1]
            self.test_mnist(test_loader, device)
            self.save_result_file()
            self.epoch += 1
            self.update_learning_rate()
            if self.epoch == self.max_epoch:
                print('Training terminated, maximum epoch reached')
            print('Epoch: ' + str(self.epoch) + ' ------------------------')
        self.global_step = 0
        self.save_csv_file()
        self.writer.close()
        print('====== Training finished =======')

    def test_mnist(self, test_loader, device):
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.view(-1, 28 * 28, 1)
            target = hf.one_hot(target, 10)
            data, target = data.to(device), target.to(device)
            self.test_step(data, target)
        self.save_test_results_epoch()

    def run_dataset(self, input_data, targets, input_data_test, targets_test):
        """ Train the network on a given dataset of size
         number of batches x batch size x input/target size x 1"""
        if not (input_data.size(0) == targets.size(0) and input_data.size(1) ==
                targets.size(1)):
            raise ValueError("InputData and Targets have not the same size")
        epoch_loss = float('inf')
        print('====== Training started =======')
        print('Epoch: ' + str(self.epoch) + ' ------------------------')
        while epoch_loss > self.threshold and self.epoch < self.max_epoch:
            for i in range(input_data.size(0)):
                data = input_data[i, :, :, :]
                target = targets[i, :, :, :]
                if i % 100 == 0:
                    print('batch: ' + str(i))
                if i % 1000 == 0:
                    if type(self.network) == InvertibleNetwork:
                        self.network.init_inverses
                        print('recomputing inverses')
                self.step(data, target)
            self.save_train_results_epoch()
            epoch_loss = self.epoch_losses[-1]
            self.test_dataset(input_data_test, targets_test)
            self.save_result_file()
            self.epoch += 1
            self.update_learning_rate()
            if self.epoch == self.max_epoch:
                print('Training terminated, maximum epoch reached')
            print('Epoch: ' + str(self.epoch) + ' ------------------------')

        self.global_step = 0
        self.save_csv_file()
        self.writer.close()
        print('====== Training finished =======')

    def test_dataset(self, input_data, targets):
        for i in range(input_data.size(0)):
            data = input_data[i, :, :, :]
            target = targets[i, :, :, :]
            self.test_step(data, target)
        self.save_test_results_epoch()

    def step(self, input_batch, targets):
        raise NotImplementedError

    def save_csv_file(self):
        self.outputfile.to_csv(self.outputfile_name)


class SGD(Optimizer):
    """ Stochastic Gradient Descend optimizer"""

    def __init__(self, network, threshold, init_learning_rate, tau=100,
                 final_learning_rate=None,
                 compute_accuracies=False, max_epoch=150,
                 outputfile_name='resultfile.csv'):
        """
        :param threshold: the optimizer will run until the network loss is
        below this threshold
        :param init_learning_rate: initial learning rate
        :param network: network to train
        :param compute_accuracies: True if the optimizer should also save the
        accuracies. Only possible with
        classification problems
        :param tau: used to update the learningrate according to
        learningrate = (1-epoch/tau)*init_learning_rate +
                    epoch/tau* final_learning_rate
        :param final_learning_rate: see tau
        :type network: Network
        """
        super().__init__(network=network, max_epoch=max_epoch,
                         compute_accuracies=compute_accuracies,
                         outputfile_name=outputfile_name)
        self.set_threshold(threshold)
        self.set_learning_rate(init_learning_rate)
        self.set_init_learning_rate(init_learning_rate)
        self.set_tau(tau)
        if final_learning_rate is None:
            self.set_final_learning_rate(0.01 * init_learning_rate)
        else:
            self.set_final_learning_rate(final_learning_rate)

    def set_init_learning_rate(self, init_learning_rate):
        if not isinstance(init_learning_rate, float):
            raise TypeError("Expecting float number for "
                            "init_learning_rate, got "
                            "{}".format(type(init_learning_rate)))
        if init_learning_rate <= 0:
            raise ValueError("Expecting strictly positive float, got "
                             "{}".format(init_learning_rate))
        self.init_learning_rate = init_learning_rate

    def set_tau(self, tau):
        if not isinstance(tau, int):
            raise TypeError("Expecting int number for tau, got"
                            " {}".format(type(tau)))
        if tau <= 0:
            raise ValueError("Expecting strictly positive integer, got "
                             "{}".format(tau))
        self.tau = tau

    def set_final_learning_rate(self, final_learning_rate):
        if not isinstance(final_learning_rate, float):
            raise TypeError("Expecting float number for final_learning_rate,"
                            " got {}".format(type(final_learning_rate)))
        if final_learning_rate <= 0:
            raise ValueError("Expecting strictly positive float, got "
                             "{}".format(final_learning_rate))
        self.final_learning_rate = final_learning_rate

    def update_learning_rate(self):
        if self.epoch <= self.tau:
            alpha = float(self.epoch) / float(self.tau)
            learning_rate = (1. - alpha) * self.init_learning_rate + \
                            alpha * self.final_learning_rate
            self.set_learning_rate(learning_rate)
        else:
            pass

    def step(self, input_batch, targets):
        """ Perform one batch optimizing step"""
        self.network.propagate_forward(input_batch)
        self.network.propagate_backward(targets)
        self.network.compute_gradients()
        self.network.update_parameters(self.learning_rate)
        self.save_results(targets)
        self.global_step += 1

class SGDInvertible(SGD):
    """ Stochastic Gradient Descent, customized for networks trained by target
    propagation with invertible layers"""

    def __init__(self, network, threshold, init_step_size, tau=100,
                 final_step_size=None, learning_rate=0.5,
                 compute_accuracies=False, max_epoch=150,
                 outputfile_name='resultfile.csv'):
        """
        :param threshold: the optimizer will run until the network loss is
        below this threshold
        :param init_learning_rate: initial learning rate
        :param network: network to train
        :param compute_accuracies: True if the optimizer should also save the
        accuracies. Only possible with
        classification problems
        :param tau: used to update the learningrate according to
        learningrate = (1-epoch/tau)*init_learning_rate +
                    epoch/tau* final_learning_rate
        :param final_learning_rate: see tau
        :type network: Network
        """
        super().__init__(network=network,threshold=threshold,
                         init_learning_rate=learning_rate,
                         tau=tau,final_learning_rate=learning_rate,
                         compute_accuracies=compute_accuracies,
                         max_epoch=max_epoch, outputfile_name=outputfile_name)
        self.set_init_step_size(init_step_size)
        self.set_step_size(init_step_size)
        self.set_tau(tau)
        if final_step_size is None:
            self.set_final_step_size(0.01 * self.init_step_size)
        else:
            self.set_final_step_size(final_step_size)

    def set_init_step_size(self, init_step_size):
        if not isinstance(init_step_size, float):
            raise TypeError("Expecting float number for "
                            "init_learning_rate, got "
                            "{}".format(type(init_step_size)))
        if init_step_size <= 0:
            raise ValueError("Expecting strictly positive float, got "
                             "{}".format(init_step_size))
        self.init_step_size = init_step_size

    def set_final_step_size(self, final_step_size):
        if not isinstance(final_step_size, float):
            raise TypeError("Expecting float number for "
                            "init_learning_rate, got "
                            "{}".format(type(final_step_size)))
        if final_step_size <= 0:
            raise ValueError("Expecting strictly positive float, got "
                             "{}".format(final_step_size))
        self.final_step_size = final_step_size

    def set_step_size(self, step_size):
        if not isinstance(step_size, float):
            raise TypeError("Expecting float number for "
                            "step_size, got "
                            "{}".format(type(step_size)))
        if step_size <= 0:
            raise ValueError("Expecting strictly positive float, got "
                             "{}".format(step_size))
        self.network.layers[-1].step_size = step_size

    def update_learning_rate(self):
        if self.epoch <= self.tau:
            alpha = float(self.epoch) / float(self.tau)
            step_size = (1. - alpha) * self.init_step_size + \
                            alpha * self.final_step_size
            self.set_step_size(step_size)
        else:
            pass



class SGDMomentum(SGD):
    """ Stochastic Gradient Descend with momentum"""

    def __init__(self, network, threshold, init_learning_rate, tau=100,
                 final_learning_rate=None,
                 compute_accuracies=False, max_epoch=150, momentum=0.5,
                 outputfile_name='resultfile.csv'):
        """
        :param momentum: Momentum value that characterizes how much of the
        previous gradients is incorporated in the
                        update.
        """
        super().__init__(network=network, threshold=threshold,
                         init_learning_rate=init_learning_rate, tau=tau,
                         final_learning_rate=final_learning_rate,
                         compute_accuracies=compute_accuracies,
                         max_epoch=max_epoch,
                         outputfile_name=outputfile_name)
        self.set_momentum(momentum)
        self.network.init_velocities()

    def set_momentum(self, momentum):
        if not isinstance(momentum, float):
            raise TypeError("Expecting float number for momentum, "
                            "got {}".format(type(momentum)))
        if not (momentum >= 0. and momentum < 1.):
            raise ValueError("Expecting momentum in [0;1), got {}".format(
                momentum))
        self.momentum = momentum

    def step(self, input_batch, targets):
        """ Perform one batch optimizing step"""
        self.network.propagate_forward(input_batch)
        self.network.propagate_backward(targets)
        self.network.compute_gradients()
        self.network.compute_gradient_velocities(self.momentum,
                                                 self.learning_rate)
        self.network.update_parameters_with_velocity()
        self.save_results(targets)
        self.global_step += 1
