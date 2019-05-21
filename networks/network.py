"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from layers.layer import Layer, InputLayer, OutputLayer, CapsuleOutputLayer


class Network(object):
    """ Network consisting of multiple layers. This class provides a range of
    methods to facilitate training of the
    networks """

    def __init__(self, layers, log=True, name=None):
        """
        :param layers: list of all the layers in the network
        :param writer: SummaryWriter object to save states of the layer
        :param log: save logs to tensorboard
        to tensorboard
        """
        self.set_name(name)
        self.set_layers(layers)
        self.writer = self.layers[0].writer
        self.set_log(log)
        self.global_step = 0

    def set_log(self, log):
        if not isinstance(log, bool):
            raise TypeError('Expecting a bool for variable log. '
                            'Got {}'.format(type(log)))
        self.log = log

    def set_layers(self, layers):
        if not isinstance(layers, list):
            raise TypeError("Expecting a list object containing all the "
                            "layers of the network")
        if len(layers) < 2:
            raise ValueError("Expecting at least 2 layers (including input "
                             "and output layer) in a network")
        # if not isinstance(layers[0], InputLayer):
        #     raise TypeError("First layer of the network should be of type"
        #                     " InputLayer")
        # if not isinstance(layers[-1], OutputLayer):
        #     raise TypeError("Last layer of the network should be of "
        #                     "type OutputLayer")
        for i in range(1, len(layers)):
            if not isinstance(layers[i], Layer):
                TypeError("All layers of the network should be of type Layer")
            if not layers[i - 1].layer_dim == layers[i].in_dim:
                raise ValueError("layer_dim should match with in_dim of "
                                 "next layer")

        self.layers = layers
        if self.name is not None:
            self.set_layer_names()

    def set_name(self, name):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("Expecting a string or None "
                                "as name for the network")
        self.name = name

    def set_layer_names(self):
        """ appends a prefix <network_name>/ to each layer name, to structure
        the tensorboard logs if multiple networks are used for an experiment
        """
        for layer in self.layers:
            layer.set_name(self.name + '/' + layer.name)

    def init_velocities(self):
        """ Initialize the gradient velocities in all the layers. Only called
        when an optimizer with momentum is used."""
        for i in range(1, len(self.layers)):
            self.layers[i].init_velocities()

    def propagate_forward(self, input_batch):
        """ Propagate the inputbatch forward through the network
        :param input_batch: Inputbatch of dimension
        batch dimension x input dimension x 1"""
        self.layers[0].set_forward_output(input_batch)
        for i in range(1, len(self.layers)):
            self.layers[i].propagate_forward(self.layers[i - 1])

    def propagate_backward(self, target):
        """ Propagate the gradient of the loss function with respect to the
        linear activation of each layer backward
        through the network
        :param target: 3D tensor of size batchdimension x class dimension x 1
        """
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not type(self.layers[-1]) == CapsuleOutputLayer:
            if not self.layers[-1].forward_output.shape == target.shape:
                raise ValueError('Expecting a tensor of dimensions: '
                                 'batchdimension x class dimension x 1.'
                                 ' Given target'
                                 'has shape' + str(target.shape))


        self.layers[-1].compute_backward_output(target)
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].propagate_backward(self.layers[i + 1])

    def compute_forward_gradients(self):
        """compute the gradient of the loss function to the
        parameters of each layer"""
        for i in range(1, len(self.layers)):
            self.layers[i].compute_forward_gradients(self.layers[i - 1])

    def compute_forward_gradient_velocities(self, momentum, learning_rate):
        """Compute the gradient velocities for each layer"""
        for i in range(1, len(self.layers)):
            self.layers[i].compute_forward_gradient_velocities(
                self.layers[i - 1],
                                                               momentum,
                                                               learning_rate)

    def compute_gradients(self):
        self.compute_forward_gradients()

    def compute_gradient_velocities(self, momentum, learning_rate):
        self.compute_forward_gradient_velocities(momentum, learning_rate)

    def update_forward_parameters_with_velocity(self):
        """ Update all the parameters of the network with the
                computed gradients velocities"""
        for i in range(1, len(self.layers)):
            self.layers[i].update_forward_parameters_with_velocity()

    def update_forward_parameters(self, learning_rate):
        """ Update all the parameters of the network with the
        computed gradients"""
        for i in range(1, len(self.layers)):
            self.layers[i].update_forward_parameters(learning_rate)

    def update_parameters(self, learning_rate):
        self.update_forward_parameters(learning_rate)

    def update_parameters_with_velocity(self):
        self.update_forward_parameters_with_velocity()

    def loss(self, target):
        """ Return the loss of each sample in the batch compared to
        the provided targets.
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        return self.layers[-1].loss(target)

    def zero_grad(self):
        """ Set all the gradients of the network to zero"""
        for layer in self.layers:
            layer.zero_grad()

    def predict(self, input_batch):
        """ Return the networks predictions on a given input batch"""
        self.propagate_forward(input_batch)
        return self.layers[-1].forward_output

    def accuracy(self, targets):
        """ Return the test accuracy of network based on the given input
        test batch and the true targets
        IMPORTANT: first you have to run self.predict(inputBatch) in order
        to save the predictions in the output
        layer.
        IMPORTANT: the accuracy can only be computed for classification
        problems, thus the last layer should be
        a softmax """
        return self.layers[-1].accuracy(targets)

    def get_output(self):
        return self.layers[-1].forward_output

    def set_global_step(self, global_step):
        self.global_step = global_step
        for layer in self.layers:
            layer.global_step = global_step

    def save_state_histograms(self, global_step):
        if self.log:
            self.set_global_step(global_step=global_step)
            for layer in self.layers:
                layer.save_state_histograms()

    def save_state(self, global_step):
        if self.log:
            self.set_global_step(global_step)
            for layer in self.layers:
                layer.save_state()
