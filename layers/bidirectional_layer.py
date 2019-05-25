"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from utils import helper_functions as hf
from utils.helper_classes import NetworkError
from layers.layer import Layer


class BidirectionalLayer(Layer):
    """ Layer in a neural network with feedforward weights as well as
    feedbackward weights."""

    def __init__(self, in_dim, layer_dim, out_dim, writer, loss_function='mse',
                 name='bidirectional_layer', debug_mode=True,
                 weight_decay=0.0, fixed=False):
        super().__init__(in_dim, layer_dim, name=name, writer=writer,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        if out_dim is not None:
            # if the layer is an outputlayer, out_dim is None
            self.set_out_dim(out_dim)
        self.init_backward_parameters()
        self.set_loss_function(loss_function)

    def set_out_dim(self, out_dim):
        if not isinstance(out_dim, int):
            raise TypeError("Expecting an integer layer dimension")
        if out_dim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.out_dim = out_dim

    def set_loss_function(self, loss_function):
        if not isinstance(loss_function, str):
            raise TypeError("Expecting a string to indicate loss function, "
                            "got {}".format(type(loss_function)))
        self.loss_function = loss_function

    def init_backward_parameters(self):
        """ Initializes the layer parameters when the layer is created.
        This method should only be used when creating
        a new layer. Use setbackwardParameters to update the parameters and
        computeGradient to update the gradients"""
        self.backward_weights = hf.get_invertible_random_matrix(self.layer_dim,
                                                                self.out_dim)
        self.backward_bias = torch.zeros(self.layer_dim, 1)
        self.backward_weights_grad = torch.zeros(self.layer_dim, self.out_dim)
        self.backward_bias_grad = torch.zeros(self.layer_dim, 1)
        self.save_initial_backward_state()

    def set_backward_parameters(self, backward_weights, backward_bias):
        if not isinstance(backward_weights, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.backward_weights")
        if not isinstance(backward_bias, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.backward_bias")
        if hf.contains_nans(backward_weights):
            raise ValueError("backward_weights contains NaNs")
        if hf.contains_nans(backward_bias):
            raise ValueError("backward_bias contains NaNs")
        if not backward_weights.shape == self.backward_weights.shape:
            raise ValueError("backward_weights has not the correct shape")
        if not backward_bias.shape == self.backward_bias.shape:
            raise ValueError("backward_bias has not the correct shape")

        self.backward_weights = backward_weights
        self.backward_bias = backward_bias

    def set_backward_gradients(self, backward_weights_grad, backward_bias_grad):
        if not isinstance(backward_weights_grad, torch.Tensor):
            raise TypeError("Expecting a tensor object "
                            "for self.backward_weights_grad")
        if not isinstance(backward_bias_grad, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.backward_bias_grad")
        if hf.contains_nans(backward_weights_grad):
            raise ValueError("backward_weights_grad contains NaNs")
        if hf.contains_nans(backward_bias_grad):
            raise ValueError("backward_bias contains NaNs")
        if not backward_weights_grad.shape == self.backward_weights_grad.shape:
            raise ValueError("backward_weights_grad has not the correct shape")
        if not backward_bias_grad.shape == self.backward_bias_grad.shape:
            raise ValueError("backward_bias_grad has not the correct shape")

        self.backward_weights_grad = backward_weights_grad
        self.backward_bias_grad = backward_bias_grad

    def set_backward_output(self, backward_output):
        if not isinstance(backward_output, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.backward_output")
        if not backward_output.size(-2) == self.layer_dim:
            raise ValueError("Expecting same dimension as layer_dim")
        if not backward_output.size(-1) == 1:
            raise ValueError("Expecting same dimension as layer_dim")
        self.backward_output = backward_output

    def backward_nonlinearity(self, linear_activation):
        """ This method should be always overwritten by the children"""
        raise NetworkError("The method backward_nonlinearity should always be "
                           "overwritten by children of Layer. Layer on itself "
                           "cannot be used in a network")

    def update_backward_parameters(self, learning_rate, upper_layer):
        """ Should be implemented by the child classes"""
        raise NetworkError('This method should be overwritten by the '
                           'child classes')

    def distance_target(self):
        """ Computes ratio of the L2 norm of the distance between the
        forward activation
        and the target activation and the L2 norm of the forward activation.
        If the batch
        contains more than one sample, the average of the distances is returned.
        """
        differences = self.forward_output - self.backward_output
        distances = torch.norm(differences.view(differences.shape[0], -1),
                               p=2, dim=1)
        forward_norm = torch.norm(self.forward_output.view(
            self.forward_output.shape[0], -1), p=2, dim=1)
        relative_distances = torch.div(distances, forward_norm)
        return torch.Tensor([torch.mean(relative_distances)])

    def save_backward_weights(self):
        weight_norm = torch.norm(self.backward_weights)
        bias_norm = torch.norm(self.backward_bias)
        # print('{} backward_weights_norm: {}'.format(self.name, weight_norm))
        self.writer.add_scalar(tag='{}/backward_weights'
                                   '_norm'.format(self.name),
                               scalar_value=weight_norm,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/backward_bias'
                                   '_norm'.format(self.name),
                               scalar_value=bias_norm,
                               global_step=self.global_step)

    def save_backward_weights_hist(self):
        self.writer.add_histogram(tag='{}/backward_weights_'
                                      'hist'.format(
            self.name),
            values=self.backward_weights,
            global_step=self.global_step)
        self.writer.add_histogram(tag='{}/backward_bias_'
                                      'hist'.format(
            self.name),
            values=self.backward_bias,
            global_step=self.global_step)

    def save_backward_activations(self):
        activations_norm = torch.norm(self.backward_output)
        self.writer.add_scalar(tag='{}/backward_activations'
                                   '_norm'.format(self.name),
                               scalar_value=activations_norm,
                               global_step=self.global_step)

    def save_backward_activations_hist(self):
        self.writer.add_histogram(tag='{}/backward_activations_'
                                      'hist'.format(
            self.name),
            values=self.backward_output,
            global_step=self.global_step)


    def save_distance_target(self):
        mean_distance = self.distance_target()
        # print('{} distance target: {}'.format(self.name, mean_distance))
        self.writer.add_scalar(tag='{}/distance_target'.format(self.name),
                               scalar_value=mean_distance,
                               global_step=self.global_step)

    def save_invertibility_test(self):
        mean_distance = self.distance_target()
        # print('{} invertibility test: {}'.format(self.name, mean_distance))
        self.writer.add_scalar(tag='{}/invertibility_distance'
                               .format(self.name),
                               scalar_value=mean_distance,
                               global_step=self.global_step)

    def save_state(self):
        """ Saves summary scalars (2-norm) of the gradients, weights and
         layer activations."""
        # Save norms
        self.save_activations()
        self.save_forward_weights()
        self.save_forward_weight_gradients()
        self.save_backward_weights()
        self.save_backward_activations()
        self.save_distance_target()

    def save_state_histograms(self):
        """ The histograms (specified by the arguments) are saved to
        tensorboard"""
        # Save histograms
        self.save_forward_weights_gradients_hist()
        self.save_forward_weights_hist()
        self.save_activations_hist()
        self.save_backward_activations_hist()
        self.save_backward_weights_hist()

    def save_initial_backward_state(self):
        self.writer.add_histogram(tag='{}/backward_weights_initial_'
                                      'hist'.format(
            self.name),
            values=self.backward_weights,
            global_step=0)
        self.writer.add_histogram(tag='{}/backward_bias_initial'
                                      'hist'.format(
            self.name),
            values=self.backward_bias,
            global_step=0)
