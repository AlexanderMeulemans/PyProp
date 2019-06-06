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
from layers.target_prop_layer import TargetPropLayer

class OriginalTPLayer(TargetPropLayer):

    def backward_nonlinearity(self, input):
        """ has to be implemented by the child class"""
        raise NetworkError('has to be implemented by the child class')

    def update_backward_parameters(self, learning_rate, upper_layer):
        nonlinear_activation = upper_layer.forward_output
        linear_activation2 = torch.matmul(self.backward_weights,
                                          nonlinear_activation) + \
                             self.backward_bias
        nonlinear_activation2 = self.backward_nonlinearity(linear_activation2)
        approx_error = nonlinear_activation2 - self.forward_output
        gradient = torch.matmul(
            self.compute_backward_vectorized_jacobian(linear_activation2)*approx_error,
                                torch.transpose(nonlinear_activation, -1, -2))
        updated_weights = (1 - learning_rate * self.weight_decay_backward) * \
                          self.backward_weights - \
                          learning_rate * torch.mean(gradient, 0)
        # updated_bias = self.backward_bias + learning_rate*torch.mean(approx_error, 0)
        updated_bias = self.backward_bias

        self.set_backward_parameters(updated_weights, updated_bias)

    def compute_backward_vectorized_jacobian(self, linear_activation, upper_layer=None):
        """ has to be implemented by the child class"""
        raise NetworkError('has to be implemented by the child class')

    def propagate_backward(self, upper_layer):
        if not isinstance(upper_layer, TargetPropLayer):
            raise TypeError("Expecting an InvertibleLayer object as argument "
                            "for "
                            "propagate_backward")
        if not upper_layer.in_dim == self.layer_dim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")
        linear_output = torch.matmul(self.backward_weights,
                                     upper_layer.backward_output) \
                        + self.backward_bias

        backward_output = self.backward_nonlinearity(
            linear_output)
        self.set_backward_output(backward_output)

    def propagate_GN_error(self, upper_layer):
        linear_activation = torch.matmul(self.backward_weights,
                                         upper_layer.forward_output) + \
                            self.backward_bias
        D_inv = self.compute_backward_vectorized_jacobian(
           linear_activation, upper_layer
        )
        self.GN_error = D_inv*torch.matmul(self.backward_weights,
                                           upper_layer.GN_error)

class OriginalTPLeakyReluLayer(OriginalTPLayer):
    """ Layer of an invertible neural network with a leaky RELU activation
    fucntion. """

    def __init__(self, negative_slope, in_dim, layer_dim,
                 out_dim, writer,
                 loss_function='mse',
                 name='invertible_leaky_ReLU_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 weight_decay_backward = 0.0,
                 fixed=False):
        super().__init__(in_dim, layer_dim, out_dim,
                         writer=writer,
                         loss_function=loss_function,
                         name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         weight_decay_backward=weight_decay_backward,
                         fixed=fixed)
        self.set_negative_slope(negative_slope)

    def set_negative_slope(self, negative_slope):
        """ Set the negative slope of the leaky ReLU activation function"""
        if not isinstance(negative_slope, float):
            raise TypeError("Expecting a float number for negative_slope, "
                            "got {}".format(type(negative_slope)))
        if negative_slope <= 0:
            raise ValueError("Expecting a strictly positive float number for "
                             "negative_slope, got {}".format(negative_slope))

        self.negative_slope = negative_slope

    def forward_nonlinearity(self, linear_activation):
        activation_function = nn.LeakyReLU(self.negative_slope)
        return activation_function(linear_activation)

    def backward_nonlinearity(self,input):
        return self.forward_nonlinearity(input)

    def compute_vectorized_jacobian(self):
        """ Compute the vectorized jacobian. The jacobian is a diagonal
        matrix, so can be represented by a vector instead of a matrix. """

        output = torch.empty(self.forward_output.shape)
        for i in range(self.forward_linear_activation.size(0)):
            for j in range(self.forward_linear_activation.size(1)):
                if self.forward_linear_activation[i, j, 0] >= 0:
                    output[i, j, 0] = 1
                else:
                    output[i, j, 0] = self.negative_slope
        return output

    def compute_inverse_vectorized_jacobian(self, linear_activation):
        output = torch.empty(linear_activation.shape)
        for i in range(linear_activation.size(0)):
            for j in range(linear_activation.size(1)):
                if linear_activation[i, j, 0] >= 0:
                    output[i, j, 0] = 1
                else:
                    output[i, j, 0] = self.negative_slope**(-1)
        return output

    def compute_backward_vectorized_jacobian(self, linear_activation, upper_layer=None):
        output = torch.empty(linear_activation.shape)
        for i in range(linear_activation.size(0)):
            for j in range(linear_activation.size(1)):
                if linear_activation[i, j, 0] >= 0:
                    output[i, j, 0] = 1
                else:
                    output[i, j, 0] = self.negative_slope
        return output

    def compute_inverse_vectorized_jacobian(self, linear_activation):
        output = torch.empty(linear_activation.shape)
        for i in range(linear_activation.size(0)):
            for j in range(self.forward_output.size(1)):
                if linear_activation[i, j, 0] >= 0:
                    output[i, j, 0] = 1
                else:
                    output[i, j, 0] = self.negative_slope ** (-1)
        return output


class OriginalTPLinearLayer(OriginalTPLayer):
    """ Layer in neural network that is purely linear and invertible"""

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def backward_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)

    def compute_backward_vectorized_jacobian(self, linear_activation, upper_layer=None):
        return torch.ones(linear_activation.shape)


class OriginalTPOutputLayer(OriginalTPLayer):
    """ Super class for the last layer of an invertible network, that will be
    trained using target propagation"""

    def __init__(self, in_dim, layer_dim, writer, step_size,
                 loss_function='mse',
                 output_loss_function='mse',
                 name='invertible_output_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 fixed=False):
        super().__init__(in_dim, layer_dim, writer=writer,
                         out_dim=None,
                         loss_function=loss_function,
                         name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        self.set_stepsize(step_size)
        self.set_output_loss_function(output_loss_function)

    def set_output_loss_function(self, output_loss_function):
        if not isinstance(output_loss_function, str):
            raise TypeError('Expecting string for output_loss_function')
        self.output_loss_function = output_loss_function

    def set_stepsize(self, step_size):
        if not isinstance(step_size, float):
            raise TypeError("Expecting float number as step_size, "
                            "got {}".format(type(step_size)))
        if step_size <= 0.:
            raise ValueError("Expecting strictly positive step_size")
        if step_size > 0.5:
            raise RuntimeWarning("Stepsize bigger then 0.5 for setting output "
                                 "target can result in unexpected behaviour")
        self.step_size = step_size

    def loss(self, target):
        """ Compute the loss with respect to the target
        :param target: 3D tensor of size batchdimension x class dimension x 1
        """
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension'
                             ' x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        if self.output_loss_function == 'crossEntropy':
            # Convert output 'class probabilities' to one class per batch sample
            #  (with highest class probability)
            target_classes = hf.prob2class(target)
            loss_function = nn.CrossEntropyLoss()
            forward_output_squeezed = torch.reshape(self.forward_output,
                                                  (self.forward_output.shape[0],
                                                   self.forward_output.shape[
                                                       1]))
            loss = loss_function(forward_output_squeezed, target_classes)
            return torch.Tensor([torch.mean(loss)])
        elif self.output_loss_function == 'mse':
            loss_function = nn.MSELoss()
            forward_output_squeezed = torch.reshape(self.forward_output,
                                                  (self.forward_output.shape[0],
                                                   self.forward_output.shape[
                                                       1]))
            target_squeezed = torch.reshape(target,
                                           (target.shape[0],
                                            target.shape[1]))
            loss = loss_function(forward_output_squeezed, target_squeezed)
            return torch.Tensor([torch.mean(loss)])

    def propagate_backward(self, upper_layer):
        """ This function should never be called for an output layer,
        the backward_output should be set based on the
        loss of the layer with compute_backward_output"""
        raise NetworkError("Propagate Backward should never be called for an "
                           "output layer, use compute_backward_output "
                           "instead")

    def init_backward_parameters(self):
        """ Outputlayer does not have backward parameters"""
        pass

    def set_backward_parameters(self, backward_weights, backward_bias):
        """ Outputlayer does not have backward parameters"""
        raise NetworkError("Outputlayer does not have backward parameters")

    def set_backward_gradients(self, backward_weights_grad, backward_bias_grad):
        """ Outputlayer does not have backward parameters"""
        raise NetworkError("Outputlayer does not have backward parameters")

    def save_state(self):
        """ Saves summary scalars (2-norm) of the gradients, weights and
                 layer activations."""
        # Save norms
        self.save_activations()
        self.save_forward_weights()
        self.save_forward_weight_gradients()
        self.save_backward_activations()
        self.save_distance_target()
        self.save_approx_error()
        self.save_approx_angle_error()

    def save_state_histograms(self):
        """ The histograms (specified by the arguments) are saved to
                tensorboard"""
        # Save histograms
        self.save_forward_weights_gradients_hist()
        self.save_forward_weights_hist()
        self.save_activations_hist()
        self.save_backward_activations_hist()


class OriginalTPLinearOutputLayer(OriginalTPOutputLayer):
    """ Invertible output layer with a linear activation function. This layer
    can so far only be combined with an mse loss
    function."""

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def backward_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)

    def compute_backward_vectorized_jacobian(self, linear_activation, upper_layer=None):
        return torch.ones(linear_activation.shape)

    def compute_inverse_vectorized_jacobian(self, linear_activation):
        return torch.ones(linear_activation.shape)

    def compute_backward_output(self, target):
        """ Compute the backward output based on a small move from the
        forward output in the direction of the negative gradient of the loss
        function."""
        if not self.output_loss_function == 'mse':
            raise NetworkError("a linear output layer can only be combined "
                               "with a mse loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension '
                             'x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        gradient = torch.mul(self.forward_output - target, 2)
        self.backward_output = self.forward_output - torch.mul(gradient,
                                                               self.step_size)

    def compute_GN_error(self, target):
        gradient = torch.mul(self.forward_output - target, 2)
        self.GN_error = torch.mul(gradient, self.step_size)
        self.real_GN_error = torch.mul(gradient, self.step_size)
        self.BP_error = gradient


class OriginalTPInputLayer(OriginalTPLayer):
    """ Input layer of the invertible neural network,
        e.g. the pixelvalues of a picture. """

    def __init__(self, layer_dim, out_dim, writer,
                 loss_function='mse',
                 name='invertible_input_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 fixed=False,
                 weight_decay_backward=0.):
        super().__init__(in_dim=None, layer_dim=layer_dim,
                         out_dim=out_dim,
                         writer=writer,
                         loss_function=loss_function,
                         name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed,
                         weight_decay_backward=weight_decay_backward)

    def init_forward_parameters(self):
        """ InputLayer has no forward parameters"""
        pass

    def init_forward_parameters_tilde(self):
        """ InputLayer has no forward parameters"""
        pass

    def backward_nonlinearity(self, input):
        return input


    def compute_backward_vectorized_jacobian(self, linear_activation, upper_layer=None):
        return torch.ones(linear_activation.shape)

    def init_velocities(self):
        """ InputLayer has no forward parameters"""
        raise RuntimeWarning("InputLayer has no forward parameters, so cannot "
                             "initialize velocities")
        pass

    def propagate_forward(self, lower_layer):
        """ This function should never be called for an input layer,
        the forward_output should be directly set
        to the input values of the network (e.g. the pixel values of a picture)
        """
        raise NetworkError("The forward_output should be directly set "
                           "to the input values of the network for "
                           "an InputLayer")

    def save_state(self):
        """ Saves summary scalars (2-norm) of the gradients, weights and
                 layer activations."""
        # Save norms
        self.save_activations()
        self.save_backward_weights()
        self.save_backward_activations()
        self.save_distance_target()

    def save_state_histograms(self):
        """ The histograms (specified by the arguments) are saved to
                tensorboard"""
        # Save histograms
        self.save_activations_hist()
        self.save_backward_activations_hist()
        self.save_backward_weights_hist()