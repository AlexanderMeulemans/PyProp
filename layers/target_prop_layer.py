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
import numpy as np

class TargetPropLayer(BidirectionalLayer):
    """ Target propagation with approximate inverses, but still the right
    form of the inverse."""
    def __init__(self, in_dim, layer_dim, out_dim, writer, loss_function='mse',
                 name='target_prop_layer', debug_mode=True,
                 weight_decay=0.0, weight_decay_backward=0.0,
                 fixed=False):
        super().__init__(in_dim, layer_dim, out_dim,
                         loss_function=loss_function,
                         name=name,
                         writer=writer,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        self.weight_decay_backward = weight_decay_backward
        self.approx_errors = torch.Tensor([])
        self.approx_error_angles = torch.Tensor([])
        self.GN_errors = torch.Tensor([])
        self.TP_errors = torch.Tensor([])
        self.GN_angles = torch.Tensor([])
        self.BP_angles = torch.Tensor([])

    def init_inverse(self, upper_layer):
        """ Initializes the backward weights to the inverse of the
        forward weights. After this
        initial inverse is computed, the sherman-morrison formula can be
        used to compute the
        inverses later in training"""
        self.backward_weights = torch.pinverse(
            upper_layer.forward_weights)

    def inverse_nonlinearity(self, input):
        """ Returns the inverse of the forward nonlinear activation function,
        performed on the given input.
        IMPORTANT: this function should always be overwritten by a child of
        InvertibleLayer, as now the forward nonlinearity is not yet specified"""
        raise NetworkError("inverse_nonlinearity should be overwritten by a "
                           "child of InvertibleLayer")

    def backward_nonlinearity(self, input, upper_layer):
        """ Take the inverse nonlinearity of the upper layer as backward
        nonlinearity."""
        return upper_layer.inverse_nonlinearity(input)

    def update_backward_parameters(self, learning_rate, upper_layer):

        noise_input = torch.randn(self.forward_output.shape)
        linear_activation = torch.matmul(upper_layer.forward_weights,
                                         noise_input) + upper_layer.forward_bias
        nonlinear_activation = upper_layer.forward_nonlinearity(linear_activation)
        nonlinear_activation2 = self.backward_nonlinearity(nonlinear_activation,
                                                           upper_layer)
        linear_activation2 = torch.matmul(self.backward_weights,
                                          nonlinear_activation2) + \
                                            self.backward_bias
        approx_error = linear_activation2 - noise_input
        gradient = torch.matmul(approx_error,
                               torch.transpose(nonlinear_activation2, -1, -2))
        updated_weights = (1-learning_rate*self.weight_decay_backward) * \
                          self.backward_weights - \
                          learning_rate*torch.mean(gradient,0)
        # updated_bias = self.backward_bias + learning_rate*torch.mean(approx_error, 0)
        updated_bias = self.backward_bias

        self.set_backward_parameters(updated_weights, updated_bias)

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

        target_inverse = self.backward_nonlinearity(
            upper_layer.backward_output, upper_layer)


        backward_output = torch.matmul(self.backward_weights,
                                      target_inverse) + self.backward_bias
        self.set_backward_output(backward_output)

    def compute_forward_gradients(self, lower_layer):
        """
        :param lower_layer: first layer upstream of the layer self
        :type lower_layer: Layer
        :return: saves the gradients of the local cost function to the layer
        parameters for all the batch samples

        """
        if self.loss_function == 'mse':
            local_loss_der = torch.mul(self.forward_output -
                                     self.backward_output, 2.)
        else:
            raise NetworkError("Expecting a mse local loss function")

        vectorized_jacobian = self.compute_vectorized_jacobian()
        u = torch.mul(vectorized_jacobian, local_loss_der)
        v = lower_layer.forward_output

        weight_gradients = torch.matmul(u, torch.transpose(v, -1, -2))

        # bias_gradients = u
        bias_gradients = torch.zeros(u.shape)
        self.set_forward_gradients(torch.mean(weight_gradients, 0), torch.mean(
            bias_gradients, 0))

    def compute_vectorized_jacobian(self):
        """ Compute the vectorized Jacobian (as the jacobian for a ridge
        nonlinearity is diagonal, it can be stored in a vector.
        IMPORTANT: this function should always be overwritten by children of
        InvertibleLayer"""
        raise NetworkError("compute_vectorized_jacobian should always be "
                           "overwritten by children of InvertibleLayer")

    def check_inverse(self, upper_layer):
        """ Check whether the computed inverse from iterative updates
        is still equal to the exact inverse. This is done by calculating the
        frobeniusnorm of W^(-1)*W - I
        :type upper_layer: InvertibleLayer
        """
        forward_weights = upper_layer.forward_weights
        forward_weights_pinv = torch.pinverse(forward_weights)
        error = self.backward_weights - forward_weights_pinv
        return torch.norm(error)

    def save_inverse_error(self, upper_layer):
        error = self.check_inverse(upper_layer)
        self.writer.add_scalar(tag='{}/inverse_error'.format(self.name),
                               scalar_value=error,
                               global_step=self.global_step)

    def propagate_GN_error(self, upper_layer):
        D_inv = self.compute_backward_vectorized_jacobian(
            upper_layer.forward_output, upper_layer
        )
        self.GN_error = torch.matmul(self.backward_weights,
                                     D_inv*upper_layer.GN_error)

    def propagate_real_GN_error(self, upper_layer):
        D_inv = upper_layer.compute_inverse_vectorized_jacobian(
            upper_layer.forward_output
        )
        weights_pinv = torch.pinverse(upper_layer.forward_weights)
        self.real_GN_error = torch.matmul(weights_pinv,
                                          D_inv*upper_layer.real_GN_error)

    def propagate_BP_error(self, upper_layer):
        D = upper_layer.compute_vectorized_jacobian()
        self.BP_error = torch.matmul(torch.transpose(upper_layer.forward_weights, -1,-2),
                                     D*upper_layer.BP_error)

    def compute_inverse_vectorized_jacobian(self, linear_activation):
        """ Should be implemented by child class"""
        raise NetworkError('Should be implemented by child class')

    def compute_approx_error(self):
        error = self.backward_output - self.forward_output + self.GN_error
        return error

    def compute_approx_angle_error(self):
        total_update = self.backward_output - self.forward_output
        GN_update = - self.GN_error
        angles = hf.get_angle(total_update, GN_update)
        return torch.tensor([torch.mean(angles)])

    def compute_GN_error_angle(self):
        total_update = self.backward_output - self.forward_output
        GN_update = - self.real_GN_error
        angles = hf.get_angle(total_update, GN_update)
        return torch.tensor([torch.mean(angles)])

    def compute_BP_error_angle(self):
        total_update = self.backward_output - self.forward_output
        BP_update = - self.BP_error
        angles = hf.get_angle(total_update, BP_update)
        return torch.tensor([torch.mean(angles)])

    def save_approx_angle_error(self):
        angle = self.compute_approx_angle_error()
        angle_GN = self.compute_GN_error_angle()
        angle_BP = self.compute_BP_error_angle()
        self.writer.add_scalar(tag='{}/approx_angle_error'.format(self.name),
                               scalar_value=angle,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/GN_angle'.format(self.name),
                               scalar_value=angle_GN,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/BP_angle'.format(self.name),
                               scalar_value=angle_BP,
                               global_step=self.global_step)

        self.approx_error_angles = torch.cat((self.approx_error_angles, angle))
        self.GN_angles = torch.cat((self.GN_angles, angle_GN))
        self.BP_angles = torch.cat((self.BP_angles, angle_BP))




    def save_approx_error(self):
        error = torch.mean(torch.norm(self.compute_approx_error(), dim=1))
        # error = torch.norm(torch.mean(self.compute_approx_error(),0))
        self.writer.add_scalar(tag='{}/approx_error'.format(self.name),
                               scalar_value=error,
                               global_step=self.global_step)
        self.approx_errors = torch.cat((self.approx_errors,
                                        torch.Tensor([error])))

    def save_GN_error(self):
        GN_error = torch.mean(torch.norm(self.GN_error, dim=1))
        self.writer.add_scalar(tag='{}/GN_error'.format(self.name),
                               scalar_value=GN_error,
                               global_step=self.global_step)
        self.GN_errors = torch.cat(((self.GN_errors,
                                     torch.Tensor([GN_error]))))

    def save_TP_error(self):
        error = self.backward_output - self.forward_output
        error = torch.mean(torch.norm(error, dim=1))
        self.writer.add_scalar(tag='{}/TP_error'.format(self.name),
                               scalar_value=error,
                               global_step=self.global_step)
        self.TP_errors = torch.cat((self.TP_errors,
                                    torch.Tensor([error])))

    def save_state(self):
        super().save_state()
        # self.save_approx_angle_error()
        # self.save_approx_error()
        # self.save_GN_error()
        # self.save_TP_error()

    def save_state_always(self):
        self.save_approx_angle_error()
        self.save_approx_error()
        self.save_GN_error()
        self.save_TP_error()


    def compute_backward_vectorized_jacobian(self, linear_activation,
                                             upper_layer):
        """ has to be implemented by the child class"""
        return upper_layer.compute_inverse_vectorized_jacobian(linear_activation)


class TargetPropLeakyReluLayer(TargetPropLayer):
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
            for j in range(linear_activation.size(1)):
                if linear_activation[i, j, 0] >= 0:
                    output[i, j, 0] = 1
                else:
                    output[i, j, 0] = self.negative_slope**(-1)
        return output


class TargetPropLinearLayer(TargetPropLayer):
    """ Layer in neural network that is purely linear and invertible"""

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def inverse_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)


class TargetPropOutputLayer(TargetPropLayer):
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


class TargetPropLinearOutputLayer(TargetPropOutputLayer):
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
        self.real_GN_error = torch.mul(gradient, self.step_size)
        self.BP_error = gradient


class TargetPropInputLayer(TargetPropLayer):
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