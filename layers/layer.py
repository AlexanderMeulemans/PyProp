"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helper_functions as hf
from utils.helper_classes import NetworkError
from tensorboardX import SummaryWriter


class Layer(object):
    """ Parent class of all occurring layers in neural networks with only
    feedforward weights. This class should not be used directly,
    only via its children"""
    # create class variable of existing layer names
    all_layer_names = []

    def __init__(self, in_dim, layer_dim, writer, name='layer',
                 debug_mode=True, weight_decay=0.0, fixed=False):
        """
        Initializes the Layer object
        :param in_dim: input dimension of the layer (equal
        to the layer dimension
        of the previous layer in the network)
        :param layer_dim: Layer dimension
        :param fixed: wether layer weights should be kept fixed
        """
        self.set_layer_dim(layer_dim)
        self.debug_mode = debug_mode
        if in_dim is not None:  # in_dim is None when layer is inputlayer
            self.set_in_dim(in_dim)
        self.set_name(name)
        self.set_writer(writer=writer)
        self.init_forward_parameters()
        self.global_step = 0  # needed for making plots with tensorboard
        self.weight_decay = weight_decay
        self.fixed = fixed


    def set_writer(self, writer):
        if not isinstance(writer, SummaryWriter):
            raise TypeError("Writer object has to be of type "
                            "SummaryWriter, now got {}".format(
                type(writer)))
        self.writer = writer

    def set_layer_dim(self, layer_dim):
        if not isinstance(layer_dim, int):
            raise TypeError("Expecting an integer layer dimension")
        if layer_dim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.layer_dim = layer_dim

    def set_in_dim(self, in_dim):
        if not isinstance(in_dim, int):
            raise TypeError("Expecting an integer layer dimension")
        if in_dim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.in_dim = in_dim

    def set_name(self, name):
        if not isinstance(name, str):
            raise TypeError("Expecting a string as name for the layer")
        self.name = name
        # if not name in self.__class__.all_layer_names:
        #     self.name = name
        #     self.__class__.all_layer_names.append(name)
        # else:
        #     new_name = name
        #     i = 1
        #     while new_name in self.__class__.all_layer_names:
        #         new_name = name + '_' + str(i)
        #         i += 1
        #     self.name = new_name
        #     self.__class__.all_layer_names.append(name)

    def set_forward_parameters(self, forward_weights, forward_bias):
        if not isinstance(forward_weights, torch.Tensor):
            raise TypeError(
                "Expecting a tensor object for self.forward_weights")
        if not isinstance(forward_bias, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forward_bias")
        if hf.contains_nans(forward_weights):
            raise ValueError("forward_weights contains NaNs")
        if hf.contains_nans(forward_bias):
            raise ValueError("forward_bias contains NaNs")
        if not forward_weights.shape == self.forward_weights.shape:
            raise ValueError("forward_weights has not the correct shape")
        if not forward_bias.shape == self.forward_bias.shape:
            raise ValueError("forward_bias has not the correct shape")

        # if torch.max(torch.abs(forward_weights)) > 1e3:
        #     print('forward_weights of {} have gone unbounded'.format(
        #         self.name))
        # if torch.max(torch.abs(forward_bias)) > 1e3:
        #     print('forwardBiases of {} have gone unbounded'.format(
        #         self.name))
        self.forward_weights = forward_weights
        self.forward_bias = forward_bias

    def set_forward_gradients(self, forward_weights_grad, forward_bias_grad):
        if not isinstance(forward_weights_grad, torch.Tensor):
            raise TypeError("Expecting a tensor object "
                            "for self.forward_weights_grad")
        if not isinstance(forward_bias_grad, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forward_bias_grad")
        if hf.contains_nans(forward_weights_grad):
            raise ValueError("forward_weights_grad contains NaNs")
        if hf.contains_nans(forward_bias_grad):
            raise ValueError("forward_bias contains NaNs")
        if not forward_weights_grad.shape == self.forward_weights_grad.shape:
            raise ValueError("forward_weights_grad has not the correct shape")
        if not forward_bias_grad.shape == self.forward_bias_grad.shape:
            raise ValueError("forward_bias_grad has not the correct shape")

        # if torch.max(torch.abs(forward_weights_grad)) > 1e3:
        #     print('forward_weights_grad of {} have gone unbounded'.format(
        #         self.name))
        # if torch.max(torch.abs(forward_bias_grad)) > 1e3:
        #     print('forwardBiasesGrad of {} have gone unbounded'.format(
        #         self.name))
        self.forward_weights_grad = forward_weights_grad
        self.forward_bias_grad = forward_bias_grad

    def set_forward_output(self, forward_output):
        if not isinstance(forward_output, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forward_output")
        if not forward_output.size(-2) == self.layer_dim:
            raise ValueError("Expecting same dimension as layer_dim")
        if not forward_output.size(-1) == 1:
            raise ValueError("Expecting same dimension as layer_dim")
        self.forward_output = forward_output

    def set_forward_linear_activation(self, forward_linear_activation):
        if not isinstance(forward_linear_activation, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forward_linear_activation")
        if not forward_linear_activation.size(-2) == self.layer_dim:
            raise ValueError("Expecting same dimension as layer_dim")
        if not forward_linear_activation.size(-1) == 1:
            raise ValueError("Expecting same dimension as layer_dim")
        self.forward_linear_activation = forward_linear_activation

    def set_backward_output(self, backward_output):
        if not isinstance(backward_output, torch.Tensor):
            raise TypeError(
                "Expecting a tensor object for self.backward_output")
        if not backward_output.size(-2) == self.layer_dim:
            raise ValueError("Expecting same dimension as layer_dim")
        if not backward_output.size(-1) == 1:
            raise ValueError("Expecting same dimension as layer_dim")
        self.backward_output = backward_output
        # if torch.max(torch.abs(backward_output)) > 1e2:
        #     print('backwardOutputs of {} have gone unbounded: {}'.format(
        #         self.name, torch.max(torch.abs(backward_output))))

    def init_forward_parameters(self):
        """ Initializes the layer parameters when the layer is created.
        This method should only be used when creating
        a new layer. Use set_forward_parameters to update the parameters and
        computeGradient to update the gradients"""
        self.forward_weights = hf.get_invertible_random_matrix(
            self.layer_dim,
            self.in_dim)
        self.forward_bias = torch.zeros(self.layer_dim, 1)
        self.forward_weights_grad = torch.zeros(self.layer_dim, self.in_dim)
        self.forward_bias_grad = torch.zeros(self.layer_dim, 1)
        #self.save_initial_state()

    def init_velocities(self):
        """ Initializes the velocities of the gradients. This should only be
        called when an optimizer with momentum
        is used, otherwise these attributes will not be used"""
        self.forward_weights_vel = torch.zeros(self.layer_dim, self.in_dim)
        self.forward_bias_vel = torch.zeros(self.layer_dim, 1)

    def set_forward_velocities(self, forward_weights_vel, forward_bias_vel):
        if not isinstance(forward_weights_vel, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forward_weights_vel")
        if not isinstance(forward_bias_vel, torch.Tensor):
            raise TypeError(
                "Expecting a tensor object for self.forward_bias_vel")
        if hf.contains_nans(forward_weights_vel):
            raise ValueError("forward_weights_vel contains NaNs")
        if hf.contains_nans(forward_bias_vel):
            raise ValueError("forward_bias_vel contains NaNs")
        if not forward_weights_vel.shape == self.forward_weights_vel.shape:
            raise ValueError("forward_weights_vel has not the correct shape")
        if not forward_bias_vel.shape == self.forward_bias_vel.shape:
            raise ValueError("forward_bias_vel has not the correct shape")

        self.forward_weights_vel = forward_weights_vel
        self.forward_bias_vel = forward_bias_vel

    def zero_grad(self):
        """ Set the gradients of the layer parameters to zero """
        self.forward_weights_grad = torch.zeros(self.layer_dim, self.in_dim)
        self.forward_bias_grad = torch.zeros(self.layer_dim, 1)

    def update_forward_parameters(self, learning_rate):
        """
        Update the forward weights and bias of the layer
        using the computed gradients.
        :param learning_rate: Learning rate of the layer
        """
        if not self.fixed:
            if not isinstance(learning_rate, float):
                raise TypeError("Expecting a float number as learning_rate")
            if learning_rate <= 0.:
                raise ValueError("Expecting a strictly positive learning_rate")

            forward_weights = (1-self.weight_decay*learning_rate) * \
                              self.forward_weights \
                              - torch.mul(self.forward_weights_grad, learning_rate)
            forward_bias = self.forward_bias \
                           - torch.mul(self.forward_bias_grad, learning_rate)
            self.set_forward_parameters(forward_weights, forward_bias)
            self.forward_learning_rate = learning_rate

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

        self.forward_input = lower_layer.forward_output
        forward_linear_activation = torch.matmul(self.forward_weights,
                                                 self.forward_input) + \
                                    self.forward_bias
        self.set_forward_linear_activation(forward_linear_activation)
        forward_output = self.forward_nonlinearity(
            self.forward_linear_activation)
        self.set_forward_output(forward_output)
        # if torch.max(torch.abs(forward_output)) > 1e3:
        #     print('forwardOutputs of {} have gone unbounded'.format(
        #         self.name))

    def forward_nonlinearity(self, linear_activation):
        """ This method should be always overwritten by the children"""
        raise NetworkError("The method forward_nonlinearity should always be "
                           "overwritten by children of Layer. Layer on itself "
                           "cannot be used in a network")

    def compute_forward_gradients(self, lower_layer):
        """
        :param lower_layer: first layer upstream of the layer self
        :type lower_layer: Layer
        :return: saves the gradients of the cost function to the layer
        parameters for all the batch samples

        """

        weight_gradients = torch.matmul(self.backward_output, torch.transpose(
            lower_layer.forward_output, -1, -2))
        bias_gradients = self.backward_output
        self.set_forward_gradients(torch.mean(weight_gradients, 0), torch.mean(
            bias_gradients, 0))

    def compute_forward_gradient_velocities(self, lower_layer, momentum,
                                            learning_rate):
        """ Used for optimizers with momentum. """
        if not isinstance(momentum, float):
            raise TypeError("Expecting float number for momentum, "
                            "got {}".format(type(momentum)))
        if not (momentum >= 0. and momentum < 1.):
            raise ValueError("Expecting momentum in [0;1), got {}".format(
                momentum))

        weight_gradients = self.forward_weights_grad
        bias_gradients = self.forward_bias_grad
        weight_velocities = torch.mul(self.forward_weights_vel, momentum) + \
                            torch.mul(weight_gradients, learning_rate)
        bias_velocities = torch.mul(self.forward_bias_vel, momentum) + \
                          torch.mul(bias_gradients, learning_rate)
        self.set_forward_velocities(weight_velocities, bias_velocities)

    def update_forward_parameters_with_velocity(self):
        """ Update the forward parameters with the gradient velocities
        computed in compute_forward_gradient_velocities"""
        if not self.fixed:
            forwardWeights = self.forward_weights \
                             - self.forward_weights_vel
            forwardBias = self.forward_bias \
                          - self.forward_bias_vel
            self.set_forward_parameters(forwardWeights, forwardBias)

    def propagate_backward(self, upper_layer):
        raise NetworkError('This method has to be overwritten by child classes')

    def save_state(self):
        """ Saves summary scalars (2-norm) of the gradients, weights and
         layer activations."""
        # Save norms
        self.save_activations()
        self.save_forward_weights()
        self.save_forward_weight_gradients()

    def save_state_histograms(self):
        """ The histograms (specified by the arguments) are saved to
        tensorboard"""
        # Save histograms
        self.save_forward_weights_gradients_hist()
        self.save_forward_weights_hist()
        self.save_activations_hist()

    def save_activations(self):
        """ Separate function to save the activations. This method will
        be used in save_state"""
        activations_norm = torch.norm(self.forward_output)
        self.writer.add_scalar(tag='{}/forward_activations'
                                   '_norm'.format(self.name),
                               scalar_value=activations_norm,
                               global_step=self.global_step)

    def save_activations_hist(self):
        self.writer.add_histogram(tag='{}/forward_activations_'
                                      'hist'.format(
            self.name),
            values=self.forward_output,
            global_step=self.global_step)

    def save_forward_weights(self):
        weight_norm = torch.norm(self.forward_weights)
        bias_norm = torch.norm(self.forward_bias)

        self.writer.add_scalar(tag='{}/forward_weights'
                                   '_norm'.format(self.name),
                               scalar_value=weight_norm,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/forward_bias'
                                   '_norm'.format(self.name),
                               scalar_value=bias_norm,
                               global_step=self.global_step)
        if self.debug_mode:
            U, S, V = torch.svd(self.forward_weights)
            s_max = S[0]
            s_min = S[-1]
            self.writer.add_scalar(
                tag='{}/forward_weights_s_max'.format(self.name),
                scalar_value=s_max,
                global_step=self.global_step)
            self.writer.add_scalar(
                tag='{}/forward_weights_s_min'.format(self.name),
                scalar_value=s_min,
                global_step=self.global_step)

    def save_forward_weight_gradients(self):
        gradient_norm = torch.norm(self.forward_weights_grad)
        gradient_bias_norm = torch.norm(self.forward_bias_grad)

        self.writer.add_scalar(tag='{}/forward_weights_gradient'
                                   '_norm'.format(self.name),
                               scalar_value=gradient_norm,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/forward_bias_gradient'
                                   '_norm'.format(self.name),
                               scalar_value=gradient_bias_norm,
                               global_step=self.global_step)

    def save_forward_weights_gradients_hist(self):
        self.writer.add_histogram(tag='{}/forward_weights_'
                                      'gradient_hist'.format(
            self.name),
            values=self.forward_weights_grad,
            global_step=self.global_step
        )
        self.writer.add_histogram(tag='{}/forward_bias_'
                                      'gradient_hist'.format(
            self.name),
            values=self.forward_bias_grad,
            global_step=self.global_step
        )

    def save_forward_weights_hist(self):
        self.writer.add_histogram(tag='{}/forward_weights_'
                                      'hist'.format(self.name),
                                  values=self.forward_weights,
                                  global_step=self.global_step)
        self.writer.add_histogram(tag='{}/forward_bias_'
                                      'hist'.format(
            self.name),
            values=self.forward_bias,
            global_step=self.global_step)

    def save_initial_state(self):
        self.writer.add_histogram(tag='{}/forward_weights_initial_'
                                      'hist'.format(
            self.name),
            values=self.forward_weights,
            global_step=0)
        self.writer.add_histogram(tag='{}/forward_bias_initial_'
                                      'hist'.format(
            self.name),
            values=self.forward_bias,
            global_step=0)


class ReluLayer(Layer):
    """ Layer of a neural network with a RELU activation function"""

    def forward_nonlinearity(self, linear_activation):
        """ Returns the nonlinear activation of the layer"""
        return F.relu(linear_activation)

    def propagate_backward(self, upper_layer):
        """
        :param upper_layer: the layer one step downstream of the layer 'self'
        :type upper_layer: Layer
        :return: saves the backwards output in self. backward_output is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upper_layer, Layer):
            raise TypeError("Expecting a Layer object as argument for "
                            "propagate_backward")
        if not upper_layer.in_dim == self.layer_dim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")

        self.backward_input = upper_layer.backward_output
        # Construct vectorized Jacobian for all batch samples.
        activation_der = torch.tensor(
            [[[1.] if self.forward_linear_activation[i, j, 0] > 0
              else [0.]
              for j in range(self.forward_linear_activation.size(1))]
             for i in range(self.forward_linear_activation.size(0))])
        backward_output = torch.mul(torch.matmul(torch.transpose(
            upper_layer.forward_weights, -1, -2), self.backward_input),
            activation_der)
        self.set_backward_output(backward_output)
        # if torch.max(torch.abs(backward_output)) > 1e3:
            # print('backwardOutputs of {} have gone unbounded'.format(
            #     self.name))
            # print('max backward_input: {}'.format(torch.max(torch.abs(
            #     self.backward_input))))
            # print('upper layer max forward weights: {}'.format(
            #     torch.max(torch.abs(upper_layer.forward_weights))
            # ))
            # print('Jacobian:')
            # print(activation_der)


class LeakyReluLayer(Layer):
    """ Layer of a neural network with a Leaky RELU activation function"""

    def __init__(self, negative_slope, in_dim, layer_dim, writer,
                 name='leaky_ReLU_layer', debug_mode=True,
                 weight_decay=0.0, fixed=False):
        super().__init__(in_dim, layer_dim, writer, name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        self.set_negative_slope(negative_slope)

    def set_negative_slope(self, negativeSlope):
        """ Set the negative slope of the leaky ReLU activation function"""
        if not isinstance(negativeSlope, float):
            raise TypeError("Expecting a float number for negative_slope, "
                            "got {}".format(type(negativeSlope)))
        if negativeSlope <= 0:
            raise ValueError("Expecting a strictly positive float number for "
                             "negative_slope, got {}".format(negativeSlope))

        self.negative_slope = negativeSlope

    def forward_nonlinearity(self, linear_activation):
        activationFunction = nn.LeakyReLU(self.negative_slope)
        return activationFunction(linear_activation)

    def propagate_backward(self, upper_layer):
        """
        :param upper_layer: the layer one step downstream of the layer 'self'
        :type upper_layer: Layer
        :return: saves the backwards output in self. backward_output is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upper_layer, Layer):
            raise TypeError("Expecting a Layer object as argument for "
                            "propagate_backward")
        if not upper_layer.in_dim == self.layer_dim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")

        self.backward_input = upper_layer.backward_output
        # Construct vectorized Jacobian for all batch samples.
        activation_der = torch.tensor(
            [[[1.] if self.forward_linear_activation[i, j, 0] > 0
              else [self.negative_slope]
              for j in range(self.forward_linear_activation.size(1))]
             for i in range(self.forward_linear_activation.size(0))])
        backward_output = torch.mul(torch.matmul(torch.transpose(
            upper_layer.forward_weights, -1, -2), self.backward_input),
            activation_der)
        self.set_backward_output(backward_output)


class SoftmaxLayer(Layer):
    """ Layer of a neural network with a Softmax activation function"""

    def forward_nonlinearity(self, linear_activation):
        """ Returns the nonlinear activation of the layer"""
        softmax = torch.nn.Softmax(1)
        return softmax(linear_activation)

    def propagate_backward(self, upper_layer):
        """
        :param upper_layer: the layer one step downstream of the layer 'self'
        :type upper_layer: Layer
        :return: saves the backwards output in self. backward_output is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upper_layer, Layer):
            raise TypeError("Expecting a Layer object as argument for  "
                            "propagate_backward")
        if not upper_layer.in_dim == self.layer_dim:
            raise ValueError("Layer sizes are not compatible for "
                             "propagating backwards")

        self.backward_input = upper_layer.backward_output
        # Construct Jacobian for all batch samples.
        softmax_activations = self.forward_output
        jacobian = torch.tensor([[[softmax_activations[i, j, 0] *
                                   (hf.kronecker(j, k) - softmax_activations[
                                       i, k, 0])
                                   for k in range(softmax_activations.size(1))]
                                  for j in range(softmax_activations.size(1))]
                                 for i in range(softmax_activations.size(0))])
        backward_output = torch.matmul(torch.transpose(jacobian, -1, -2),
                                      torch.matmul(torch.transpose(
                                          upper_layer.forward_weights, -1, -2)
                                          , self.backward_input))
        self.set_backward_output(backward_output)


class LinearLayer(Layer):
    """ Layer of a neural network with a linear activation function"""

    def forward_nonlinearity(self, linear_activation):
        """ Returns the nonlinear activation of the layer"""
        return linear_activation

    def propagate_backward(self, upper_layer):
        """
        :param upper_layer: the layer one step downstream of the layer 'self'
        :type upper_layer: Layer
        :return: saves the backwards output in self. backward_output is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upper_layer, Layer):
            raise TypeError("Expecting a Layer object as "
                            "argument for propagate_backward")
        if not upper_layer.in_dim == self.layer_dim:
            raise ValueError("Layer sizes are not compatible "
                             "for propagating backwards")

        self.backward_input = upper_layer.backward_output
        backward_output = torch.matmul(torch.transpose(
            upper_layer.forward_weights, -1, -2), self.backward_input)
        self.set_backward_output(backward_output)


class InputLayer(Layer):
    """ Input layer of the neural network,
    e.g. the pixelvalues of a picture. """

    def __init__(self, layer_dim, writer, name='input_layer',
                 debug_mode=True, weight_decay=0.0, fixed=False):
        """ InputLayer has only a layer_dim and a
        forward activation that can be set,
         no input dimension nor parameters"""
        super().__init__(in_dim=None, layer_dim=layer_dim, writer=writer,
                         name=name, debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)

    def propagate_forward(self, lower_layer):
        """ This function should never be called for an input layer,
        the forward_output should be directly set
        to the input values of the network (e.g. the pixel values of a picture)
        """
        raise NetworkError("The forward_output should be directly set "
                           "to the input values of the network for "
                           "an InputLayer")

    def propagate_backward(self, upper_layer):
        """ This function should never be called for an input layer,
        there is no point in having a backward output
         here, as this layer has no parameters to update"""
        raise NetworkError("Propagate Backward should never be called for "
                           "an input layer, there is no point in having "
                           "a backward output here, as this layer has no "
                           "parameters to update")

    def init_forward_parameters(self):
        """ InputLayer has no forward parameters"""
        pass

    def init_velocities(self):
        """ InputLayer has no forward parameters"""
        raise RuntimeWarning("InputLayer has no forward parameters, so cannot "
                             "initialize velocities")

    def save_state(self):
        self.save_activations()

    def save_state_histograms(self):
        self.save_activations_hist()


class OutputLayer(Layer):
    """" Super class for the last layer of a network.
    This layer has a loss as extra attribute and some extra
    methods as explained below. """

    def __init__(self, in_dim, layer_dim, loss_function, writer,
                 name='output_layer', debug_mode=True,
                 weight_decay=0.0, fixed=False):
        """
        :param in_dim: input dimension of the layer,
        equal to the layer dimension of the second last layer in the network
        :param layer_dim: Layer dimension
        :param loss: string indicating which loss function is used to
        compute the network loss
        """
        super().__init__(in_dim, layer_dim, writer, name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        self.set_loss_function(loss_function)

    def set_loss_function(self, loss_function):
        if not isinstance(loss_function, str):
            raise TypeError('Expecting a string as indicator'
                            ' for the loss function')
        if not (loss_function == 'mse' or loss_function == 'crossEntropy'
                or loss_function == 'capsule_loss'):
            raise NetworkError('Expecting an mse or crossEntropy loss')
        self.loss_function = loss_function

    def loss(self, target):
        """
        Should be overwritten by its children
        """
        raise NotImplementedError


    def propagate_backward(self, upper_layer):
        """ This function should never be called for an output layer,
        the backward_output should be set based on the
        loss of the layer with compute_backward_output"""
        raise NetworkError("Propagate Backward should never be called for an "
                           "output layer, use compute_backward_output "
                           "instead")


class ClassificationOutputLayer(OutputLayer):
    """
    Parent class for all output layers used for classification.
    """
    def propagate_forward(self, lower_layer):
        """ Normal forward propagation, but on top of that, save the predicted
        classes in self."""
        super().propagate_forward(lower_layer)
        self.predicted_classes = hf.prob2class(self.forward_output)

    def accuracy(self, target):
        """ Compute the accuracy if the network predictions with respect to
        the given true targets.
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not isinstance(self, CapsuleOutputLayer):
            if not self.forward_output.shape == target.shape:
                raise ValueError('Expecting a tensor of dimensions: batchdimension'
                                 ' x class dimension x 1. Given target'
                                 'has shape' + str(target.shape))
        return hf.accuracy(self.predicted_classes, hf.prob2class(target))


class SoftmaxOutputLayer(ClassificationOutputLayer):
    """ Output layer with a softmax activation function. This layer
    should always be combined with a crossEntropy
    loss."""

    def forward_nonlinearity(self, linear_activation):
        """ Returns the nonlinear activation of the layer"""
        softmax = torch.nn.Softmax(1)
        return softmax(linear_activation)

    def compute_backward_output(self, target):
        """ Compute the backward output based on the derivative of the loss
        to the linear activation of this layer
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        if not self.loss_function == 'crossEntropy':
            raise NetworkError("a softmax output layer should always be "
                               "combined with a cross entropy loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: '
                             'batchdimension x class dimension x 1.'
                             ' Given target'
                             'has shape' + str(target.shape))

        backward_output = self.forward_output - target
        self.set_backward_output(backward_output)


    def loss(self, target):
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension'
                             ' x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        if self.loss_function == 'crossEntropy':
            # Convert output 'class probabilities' to one class per batch sample
            #  (with highest class probability)
            target_classes = hf.prob2class(target)
            loss_function = nn.CrossEntropyLoss()
            forward_output_squeezed = torch.reshape(self.forward_output,
                                                  (self.forward_output.shape[0],
                                                   self.forward_output.shape[
                                                       1]))
            loss = loss_function(forward_output_squeezed, target_classes)
            return torch.Tensor([torch.mean(loss)])  # .numpy()

        else:
            raise NetworkError('Only crossEntropy loss defined for a soft max'
                               'output layer, got {}'.format(
                self.loss_function))


class LinearOutputLayer(OutputLayer):
    """ Output layer with a linear activation function. This layer can so far
    only be combined with an mse loss
    function."""

    def forward_nonlinearity(self, linear_activation):
        """ Returns the nonlinear activation of the layer"""
        return linear_activation

    def compute_backward_output(self, target):
        """ Compute the backward output based on the derivative of the loss to
        the linear activation of this layer"""
        if not self.loss_function == 'mse':
            raise NetworkError("a linear output layer can only be combined "
                               "with a mse loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension '
                             'x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        backward_output = 2 * (self.forward_output - target)
        self.set_backward_output(backward_output)
        # if torch.max(torch.abs(backward_output)) > 1e3:
        #     print('backwardOutputs of {} have gone unbounded: {}'.format(
        #         self.name, torch.max(torch.abs(backward_output))))

    def loss(self, target):
        """ Compute the loss with respect to the target
                :param target: 3D tensor of size
                 batchdimension x class dimension x 1
                """
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension'
                             ' x class dimension x 1. Given target'
                             'has shape' + str(target.shape))

        if self.loss_function == 'mse':
            loss_function = nn.MSELoss()
            forward_output_squeezed = torch.reshape(self.forward_output,
                                                    (self.forward_output.shape[
                                                         0],
                                                     self.forward_output.shape[
                                                         1]))
            target_squeezed = torch.reshape(target,
                                            (target.shape[0],
                                             target.shape[1]))
            loss = loss_function(forward_output_squeezed, target_squeezed)
            loss = torch.Tensor([torch.mean(loss)])
            # if loss > 1e2:
            #     # print('loss is bigger than 100. Loss: {}'.format(loss))
            #     # print('max difference: {}'.format(torch.max(torch.abs(
            #     #     forward_output_squeezed - target_squeezed
            #     # ))))

            return loss

        else:
            raise NetworkError('Only mse loss function defined for a linear'
                               'output layer, got {}'.format(
                self.loss_function))


class CapsuleOutputLayer(ClassificationOutputLayer):
    def __init__(self, in_dim, layer_dim, nb_classes, writer,
                 loss_function='capsule_loss',
                 name='invertible_capsule_output_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 fixed=False):
        super().__init__(in_dim, layer_dim,
                         writer=writer,
                         loss_function=loss_function,
                         name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        self.set_nb_classes(nb_classes)
        self.set_capsule_indices()
        self.m_plus = 0.9
        self.m_min = 0.1
        self.l = 0.5

    def set_nb_classes(self, nb_classes):
        if not isinstance(nb_classes, int):
            raise TypeError('expecting an integer type for nb_classes, '
                            'got {}'.format(type(nb_classes)))
        if nb_classes <= 0:
            raise ValueError('expecting positive integer for nb_classes,'
                             'got {}'.format(nb_classes))
        self.nb_classes = nb_classes

    def set_capsule_indices(self):
        if not self.layer_dim % self.nb_classes == 0:
            print('Warning: number of output neurons is not divisible by'
                  'number of classes. Capsules of unequal lenght are used.')
        self.excess = self.layer_dim % self.nb_classes
        self.capsule_base_size = int(self.layer_dim / self.nb_classes)
        if self.excess == 0:
            self.capsule_size = self.capsule_base_size
        else:
            self.capsule_size = self.capsule_base_size + 1
        self.capsule_indices = {}
        start = 0
        for capsule in range(self.nb_classes):
            # capsules with index below excess will have 1 element extra
            if capsule < self.excess:
                stop = start + self.capsule_base_size + 1
            else:
                stop = start + self.capsule_base_size
            self.capsule_indices[capsule] = (start, stop)
            start = stop

    def propagate_forward(self, lower_layer):
        """ Normal forward propagation, but on top of that, save the predicted
        classes in self."""
        super(ClassificationOutputLayer, self).propagate_forward(lower_layer)
        self.compute_capsules()
        self.predicted_classes = hf.prob2class(self.capsule_squashed)

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def compute_capsules(self):
        linear_activation = self.forward_linear_activation
        self.capsule_magnitudes = torch.empty((linear_activation.shape[0],
                                               self.nb_classes, 1))
        self.capsules = torch.zeros((linear_activation.shape[0],
                                     self.nb_classes,
                                     self.capsule_size))
        for k in range(self.nb_classes):
            if k < self.excess:
                self.capsules[:, k, 0:self.capsule_size] = linear_activation[:,
                                                           self.capsule_indices[
                                                               k][0]:
                                                           self.capsule_indices[
                                                               k][1], 0]
            else:
                self.capsules[:, k, 0:self.capsule_base_size] = \
                    linear_activation[:, self.capsule_indices[k][0]:
                                         self.capsule_indices[k][1], 0]
            self.capsule_magnitudes[:, k, 0] = torch.norm(
                self.capsules[:, k, :], dim=1)

        self.capsule_squashed = self.capsule_magnitudes ** 2 / (
                1 + self.capsule_magnitudes ** 2)

    def loss(self, target):
        if self.loss_function == 'capsule_loss':
            # see Hinton - Dynamic routing between capsules
            l = self.l
            m_plus = self.m_plus
            m_min = self.m_min
            L_k = target * \
                  torch.max(torch.stack([m_plus - self.capsule_squashed,
                                         torch.zeros(
                                             self.capsule_squashed.shape)]),
                            dim=0)[0] ** 2 + \
                  l * (1 - target) * \
                  torch.max(torch.stack([self.capsule_squashed - m_min,
                                         torch.zeros(
                                             self.capsule_squashed.shape)]),
                            dim=0)[0] ** 2
            loss = torch.sum(L_k, dim=1)
            loss = torch.Tensor([torch.mean(loss)])
            return loss
        else:
            raise NetworkError('Only capsule_loss is defined for a capsule'
                               'output layer, got {}'.format(
                self.loss_function))

    def compute_backward_output(self, target):
        """ Compute the backward output based on the derivative of the loss to
        the linear activation of this layer"""
        if not self.loss_function == 'capsule_loss':
            raise NetworkError("Only capsule_loss is defined for a capsule"
                               "output layer, got {}".format(
                self.loss_function))
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.capsule_squashed.shape == target.shape:
            raise ValueError(
                'Expecting a tensor of dimensions: batchdimension '
                'x class dimension x 1. Given target'
                'has shape' + str(target.shape))
        m_min = self.m_min
        m_plus = self.m_plus
        l = self.l
        Lk_vk = -2 * target * \
                torch.max(torch.stack([m_plus - self.capsule_squashed,
                                       torch.zeros(
                                           self.capsule_squashed.shape)]),
                          dim=0)[0] + \
                2 * l * (1 - target) * \
                torch.max(torch.stack([self.capsule_squashed \
                                       - m_min,
                                       torch.zeros(
                                           self.capsule_squashed.shape)]),
                          dim=0)[0]
        vk_sk = 1 / ((
                             1 + self.capsule_magnitudes ** 2) ** 2) * \
                2 * self.capsules
        backward_output = torch.empty(self.forward_linear_activation.shape)
        for k in range(self.nb_classes):
            start = self.capsule_indices[k][0]
            stop = self.capsule_indices[k][1]

            if k < self.excess:
                backward_output[:, start:stop, 0] = Lk_vk[:, k, :] * vk_sk[:, k,
                                                                     :]
            else:
                backward_output[:, start:stop, 0] = Lk_vk[:, k,
                                                    0:self.capsule_base_size] * \
                                                    vk_sk[:, k,
                                                    0:self.capsule_base_size]

        self.set_backward_output(backward_output)

    def init_forward_parameters(self):
        """ Initializes the layer parameters when the layer is created.
                This method should only be used when creating
                a new layer. Use set_forward_parameters to update the parameters and
                computeGradient to update the gradients"""
        super().init_forward_parameters()
        # self.forward_weights = self.forward_weights / float(self.layer_dim)**0.5








