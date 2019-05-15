"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
import torch.nn as nn
from utils import helper_functions as hf
from utils.helper_classes import NetworkError, NotImplementedError
from layers.layer import Layer
from layers.bidirectional_layer import BidirectionalLayer

class TargetPropLayer(BidirectionalLayer):
    """ Target propagation with approximate inverses, but still the right
    form of the inverse."""
    def __init__(self, in_dim, layer_dim, out_dim, writer, loss_function='mse',
                 name='target_prop_layer', debug_mode=True,
                 weight_decay=0.0, fixed=False):
        super().__init__(in_dim, layer_dim, out_dim,
                         loss_function=loss_function,
                         name=name,
                         writer=writer,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)

    def init_inverse(self, upper_layer):
        """ Initializes the backward weights to the inverse of the
        forward weights. After this
        initial inverse is computed, the sherman-morrison formula can be
        used to compute the
        inverses later in training"""
        self.backward_weights = torch.inverse(
            torch.cat((upper_layer.forward_weights,
                       upper_layer.forward_weights_tilde), 0))
        self.backward_bias = - torch.cat((upper_layer.forward_bias,
                                          upper_layer.forward_bias_tilde), 0)

    def inverse_nonlinearity(self, input):
        """ Returns the inverse of the forward nonlinear activation function,
        performed on the given input.
        IMPORTANT: this function should always be overwritten by a child of
        InvertibleLayer, as now the forward nonlinearity is not yet specified"""
        raise NetworkError("inverse_nonlinearity should be overwritten by a "
                           "child of InvertibleLayer")

    def update_backward_parameters(self, learning_rate, upper_layer):

        noise_input = torch.randn(self.forward_output.shape)
        linear_activation = torch.matmul(upper_layer.forward_weights,
                                         noise_input) + upper_layer.forward_bias
        nonlinear_activation = upper_layer.forward_nonlinearity(linear_activation)
        nonlinear_activation2 = self.inverse_nonlinearity(nonlinear_activation)
        linear_activation2 = torch.matmul(self.backward_weights,
                                          nonlinear_activation2) + \
                                            self.backward_bias
        approx_error = linear_activation2 - noise_input
        gradient =torch.matmul(approx_error,
                               torch.transpose(nonlinear_activation2, -1, -2))
        updated_weights = self.backward_weights + learning_rate*torch.mean(gradient,0)
        updated_bias = self.backward_bias + learning_rate*torch.mean(approx_error, 0)

        self.set_backward_parameters(updated_weights, updated_bias)