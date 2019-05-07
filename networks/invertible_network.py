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


class InvertibleNetwork(BidirectionalNetwork):
    """ Invertible Network consisting of multiple invertible layers. This class
        provides a range of methods to facilitate training of the networks """

    def __init__(self, layers, log=True, name=None, debug_mode=False):
        super().__init__(layers=layers, log=log, name=name)
        self.init_inverses()
        self.debug_mode = debug_mode

    def set_layers(self, layers):
        if not isinstance(layers, list):
            raise TypeError("Expecting a list object containing all the "
                            "layers of the network")
        if len(layers) < 2:
            raise ValueError("Expecting at least 2 layers (including input "
                             "and output layer) in a network")
        if not isinstance(layers[0], InvertibleInputLayer):
            raise TypeError("First layer of the network should be of type"
                            " InvertibleInputLayer")
        if not isinstance(layers[-1], InvertibleOutputLayer):
            raise TypeError("Last layer of the network should be of "
                            "type InvertibleOutputLayer")
        for i in range(1, len(layers)):
            if not isinstance(layers[i], InvertibleLayer):
                TypeError("All layers of the network should be of type "
                          "InvertibleLayer")
            if not layers[i - 1].layer_dim == layers[i].in_dim:
                raise ValueError("layer_dim should match with in_dim of "
                                 "next layer")
            if not layers[i - 1].out_dim == layers[i].layer_dim:
                raise ValueError(
                    "outputDim should match with layer_dim of next "
                    "layer")

        self.layers = layers

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

    def save_sherman_morrison(self):
        if self.log:
            for i in range(len(self.layers)-1):
                self.layers[i].save_sherman_morrison()

    def save_state(self, global_step):
        super().save_state(global_step)
        if self.log:
            self.save_inverse_error()
            self.save_sherman_morrison()


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

    def compute_GN_targets(self):
        Jtot = self.compute_total_jacobian()
        g = self.get_output_gradient()
        J_pinverse = torch.pinverse(Jtot, rcond=1e-6)
        htot = torch.matmul(J_pinverse, g)
        return htot

    def compute_total_jacobian(self):
        rows = self.layers[-1].layer_dim
        cols = 0
        for i in range(1,len(self.layers)-1):
            cols += self.layers[i].layer_dim

        J_tot = torch.empty(rows, cols)
        J = torch.eye(rows, rows)
        start = 0
        for i in range(len(self.layers) - 1,1,-1):
            Di = self.layers[i].compute_vectorized_jacobian()
            Di = Di.squeeze(0)
            Ji = Di*self.layers[i].forward_weights
            J = torch.matmul(J,Ji)
            size = self.layers[i].layer_dim
            J_tot[:,start:start+size] = J
            start = start+size
        return J_tot

    def get_output_gradient(self):
        gradient = self.layers[-1].forward_output-self.layers[-1].backward_output
        return gradient



