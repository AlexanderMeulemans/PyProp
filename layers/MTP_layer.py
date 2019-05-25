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

class MTPLayer(TargetPropLayer):
    """ Modified target propagation layer"""

    def propagate_forward(self, lower_layer):
        """
        :param lower_layer: The first layer upstream of the layer 'self'
        :type lower_layer: Layer
        :return saves the computed output of the layer to self.forward_output.
                forward_output is a 3D tensor of size
                batchDimension x layerDimension x 1
        """
        if not isinstance(lower_layer, Layer):
            raise TypeError("Expecting a Layer object as "
                            "argument for propagate_forward")
        if not lower_layer.layer_dim == self.in_dim:
            raise ValueError("Layer sizes are not compatible for "
                             "propagating forward")

        if isinstance(lower_layer, MTPInputLayer):
            forward_input = lower_layer.forward_output
        else:
            forward_input = lower_layer.forward_output_batchnorm
        self.forward_linear_activation = torch.matmul(self.forward_weights,
                                                 forward_input) + \
                                    self.forward_bias
        forward_output = self.forward_nonlinearity(
            self.forward_linear_activation)
        self.mu = torch.mean(forward_output, dim=0)
        self.sigma = torch.std(forward_output, dim=0)
        self.set_forward_output(forward_output)
        self.forward_output_batchnorm = 1./self.sigma*(forward_output-self.mu)


    def propagate_backward(self, upper_layer):
        """Propagate the target signal from the upper layer to the current
        layer (self)
        :type upper_layer: InvertibleLayer
        """
        if not isinstance(upper_layer, TargetPropLayer):
            raise TypeError("Expecting an InvertibleLayer object as argument "
                            "for "
                            "propagate_backward")
        if not upper_layer.in_dim == self.layer_dim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")
        backward_input = upper_layer.backward_output
        target_inverse = self.backward_nonlinearity(
            backward_input, upper_layer)


        backward_output = torch.matmul(self.backward_weights,
                                      target_inverse) + self.backward_bias
        backward_output = self.sigma*backward_output + self.mu
        self.set_backward_output(backward_output)

    # def compute_forward_gradients(self, lower_layer):
    #     """
    #     :param lower_layer: first layer upstream of the layer self
    #     :type lower_layer: Layer
    #     :return: saves the gradients of the local cost function to the layer
    #     parameters for all the batch samples
    #
    #     """
    #     if self.loss_function == 'mse':
    #         local_loss_der = torch.mul(self.forward_output -
    #                                  self.backward_output, 2.)
    #     else:
    #         raise NetworkError("Expecting a mse local loss function")
    #
    #     vectorized_jacobian = self.compute_vectorized_jacobian()
    #     u = torch.mul(vectorized_jacobian**(-1), local_loss_der)
    #     v = lower_layer.forward_output
    #
    #     weight_gradients = torch.matmul(u, torch.transpose(v, -1, -2))
    #
    #     # bias_gradients = u
    #     bias_gradients = torch.zeros(u.shape)
    #     self.set_forward_gradients(torch.mean(weight_gradients, 0), torch.mean(
    #         bias_gradients, 0))

class MTPLeakyReluLayer(MTPLayer):
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

    def inverse_nonlinearity(self, input):
        """ perform the inverse of the forward nonlinearity on the given
        input. """
        output = torch.empty(input.shape)
        for i in range(input.size(0)):
            for j in range(input.size(1)):
                for k in range(input.size(2)):
                    if input[i, j, k] >= 0:
                        output[i, j, k] = input[i, j, k]
                    else:
                        output[i, j, k] = input[i, j, k] / self.negative_slope
        return output

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
            for j in range(self.forward_output.size(1)):
                if linear_activation[i, j, 0] >= 0:
                    output[i, j, 0] = 1
                else:
                    output[i, j, 0] = self.negative_slope**(-1)
        return output


class MTPLinearLayer(MTPLayer):
    """ Layer in neural network that is purely linear and invertible"""

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def inverse_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)


class MTPOutputLayer(MTPLayer):
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

    def propagate_forward(self, lower_layer):
        """
        Don't do batch normalization in the output layer
        :param lower_layer: The first layer upstream of the layer 'self'
        :type lower_layer: Layer
        :return saves the computed output of the layer to self.forward_output.
                forward_output is a 3D tensor of size
                batchDimension x layerDimension x 1
        """
        if not isinstance(lower_layer, Layer):
            raise TypeError("Expecting a Layer object as "
                            "argument for propagate_forward")
        if not lower_layer.layer_dim == self.in_dim:
            raise ValueError("Layer sizes are not compatible for "
                             "propagating forward")

        forward_input = lower_layer.forward_output
        self.forward_linear_activation = torch.matmul(self.forward_weights,
                                                 forward_input) + \
                                    self.forward_bias
        forward_output = self.forward_nonlinearity(
            self.forward_linear_activation)
        self.mu = torch.ones((forward_output.shape[1], forward_output.shape[2]))
        self.sigma = torch.ones((forward_output.shape[1], forward_output.shape[2]))
        self.set_forward_output(forward_output)

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


class MTPLinearOutputLayer(MTPOutputLayer):
    """ Invertible output layer with a linear activation function. This layer
    can so far only be combined with an mse loss
    function."""

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def inverse_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)

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


class MTPInputLayer(MTPLayer):
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


    def propagate_backward(self, upper_layer):
        self.mu = torch.ones((self.forward_output.shape[1],
                              self.forward_output.shape[2]))
        self.sigma = torch.ones(
            (self.forward_output.shape[1], self.forward_output.shape[2]))
        super().propagate_backward(upper_layer)
