"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from layers.invertible_layer import InvertibleInputLayer, \
    InvertibleLayer, InvertibleOutputLayer
from networks.bidirectional_network import BidirectionalNetwork
import torch
import utils.helper_functions as hf
import numpy as np

class TargetPropNetwork(BidirectionalNetwork):
    def __init__(self, layers, log=True, name=None, debug_mode=False,
                 randomize=False):
        super().__init__(layers=layers, log=log, name=name)
        self.init_inverses()
        self.debug_mode = debug_mode
        self.randomize = randomize
        self.random_layers = np.array([])
        if randomize==True:
            self.random_layer = np.random.choice(len(layers) - 1, 1)[0]+1

    def init_inverses(self):
        """ Initialize the backward weights of all layers to the inverse of
        the forward weights of
        the layer on top."""
        for i in range(0, len(self.layers) - 1):
            self.layers[i].init_inverse(self.layers[i + 1])

    def save_inverse_error(self):
        if self.log:
            for i in range(0, len(self.layers) - 1):
                self.layers[i].save_inverse_error(self.layers[i + 1])

    def save_state(self, global_step):
        if self.log:
            self.set_global_step(global_step)
            self.save_inverse_error()
            if self.randomize:
                self.layers[self.random_layer].save_state()
            else:
                for layer in self.layers:
                    layer.save_state()
                self.save_angle_GN_block_approx()

    def test_invertibility(self, input_batch):
        """ Propagate an input batch forward and backward, and compute the error
        of the inversions (backpropagated targets should be equal to forward
        activations"""
        self.propagate_forward(input_batch)
        self.custom_propagate_backward(self.layers[-1].forward_output)
        for layer in self.layers:
            layer.save_invertibility_test()

    def custom_propagate_backward(self, backward_input):
        """
        Propagate directly the given backward_input backwards through the
        network instead of computing the output target value with
        compute_backward_output().
        """
        self.layers[-1].set_backward_output(backward_input)
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].propagate_backward(self.layers[i + 1])

    def save_state_histograms(self, global_step):
        """ Also perform an invertibiltiy test at the end of each batch
        if debug mode is on True"""
        super().save_state_histograms(global_step)
        if self.log:
            self.test_invertibility(self.layers[0].forward_output)
            self.save_random_layers()

    def save_angle_GN_block_approx(self):
        if self.log:
            angle = self.get_angle_GN_block_approx()
            self.writer.add_scalar(tag='network/angle_'
                                       'GN_blockapprox',
                                   scalar_value=angle,
                                   global_step=self.global_step)

    def save_random_layers(self):
        if self.log:
            if self.randomize:
                self.writer.add_histogram(tag='network/layer_updates_'
                                              'hist',
                    values=self.random_layers,
                    global_step=self.global_step)

    def get_angle_GN_block_approx(self):
        h_GN = self.compute_GN_targets()
        h_TP = self.get_activation_update()
        angle = hf.get_angle(h_GN,h_TP)
        return angle

    def compute_GN_targets(self):
        Jtot = self.compute_total_jacobian()
        g = self.get_output_gradient()
        J_pinverse = torch.pinverse(Jtot, rcond=1e-6)
        htot = torch.matmul(J_pinverse, -g)
        return htot

    def compute_total_jacobian(self):
        rows = self.layers[-1].layer_dim
        cols = 0
        for i in range(1,len(self.layers)-1):
            cols += self.layers[i].layer_dim

        J_tot = torch.empty(rows, cols)
        J = torch.eye(rows, rows)
        end = cols
        for i in range(len(self.layers) - 1,1,-1):
            Di = self.layers[i].compute_vectorized_jacobian()
            Di = Di.squeeze(0)
            Ji = Di*self.layers[i].forward_weights
            J = torch.matmul(J,Ji)
            size = self.layers[i].layer_dim
            J_tot[:,end-size:end] = J
            end = end-size
        return J_tot

    def get_output_gradient(self):
        gradient = self.layers[-1].forward_output-self.layers[-1].backward_output
        return gradient

    def get_activation_update(self):
        total_length = 0
        for i in range(1,len(self.layers)-1):
            total_length += self.layers[i].layer_dim

        h_update = torch.empty(self.layers[0].forward_output.shape[0],
                               total_length, 1)
        start = 0
        for i in range(1, len(self.layers)-1):
            stop = start + self.layers[i].layer_dim
            h_update[:,start:stop,:] = self.layers[i].backward_output - \
                self.layers[i].forward_output
            start = stop

        return h_update

    def update_backward_parameters(self, learning_rate):
        """ Update all the parameters of the network with the
        computed gradients"""
        if self.randomize == True:
            i = self.random_layer - 1
            self.layers[i].update_backward_parameters(learning_rate,
                                                      self.layers[i + 1])
        else:
            for i in range(0, len(self.layers) - 1):
                self.layers[i].update_backward_parameters(learning_rate,
                                                          self.layers[i + 1])

    def update_forward_parameters(self, learning_rate):
        """ Update all the parameters of the network with the
        computed gradients"""
        if self.randomize == True:
            self.random_layer = np.random.choice(len(self.layers) - 1, 1)[0] + 1
            self.random_layers = np.append(
                self.random_layers, self.random_layer)
            i = self.random_layer
            self.layers[i].update_forward_parameters(learning_rate)
        else:
            for i in range(1, len(self.layers)):
                self.layers[i].update_forward_parameters(learning_rate)