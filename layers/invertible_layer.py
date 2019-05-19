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


class InvertibleLayer(BidirectionalLayer):
    """ Layer that is invertible to make it able to propagate exact targets."""

    def __init__(self, in_dim, layer_dim, out_dim, writer, loss_function='mse',
                 name='invertible_layer', epsilon=0.5, debug_mode=True,
                 weight_decay=0.0, fixed=False):
        if in_dim is not None:
            if in_dim < layer_dim:
                raise ValueError(
                    "Expecting an input size bigger or equal to the "
                    "layer size")
        if out_dim is not None:
            if layer_dim < out_dim:
                raise ValueError(
                    "Expecting a layer size bigger or equal to the "
                    "output size")
        super().__init__(in_dim, layer_dim, out_dim,
                         loss_function=loss_function,
                         name=name,
                         writer=writer,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        self.init_forward_parameters_tilde()
        self.set_epsilon(epsilon)

    def init_forward_parameters_tilde(self):
        """ Initializes the layer parameters that connect the current layer
        with the random fixed features of the next layer
        when the layer is created.
        This method should only be used when creating
        a new layer. These parameters should remain fixed"""
        self.forward_weights_tilde = torch.randn(self.in_dim - self.layer_dim,
                                                 self.in_dim)
        self.forward_bias_tilde = torch.zeros(self.in_dim - self.layer_dim, 1)

    def init_backward_parameters(self):
        """ Initializes the layer parameters when the layer is created.
        This method should only be used when creating
        a new layer. Use setbackwardParameters to update the parameters and
        computeGradient to update the gradients"""
        self.backward_weights = torch.empty(self.layer_dim, self.layer_dim)
        self.backward_bias = torch.zeros(self.layer_dim, 1)

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

    # def initForwardParametersBar(self):
    #     """ Concatenates the forward_weights with the forward_weights_tilde to
    #     create a square matrix ForwardWeightsBar. Similarly concatenates the
    #     two biases. """
    #     self.forwardWeightsBar = torch.cat((self.forward_weights,
    #                                         self.forward_weights_tilde), 0)
    #     self.forwardBiasBar = torch.cat((self.forward_bias,
    #                                      self.forward_bias_tilde), 0)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_forward_output_tilde(self, forward_output_tilde):
        if not isinstance(forward_output_tilde, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forward_output_tilde")
        if forward_output_tilde.size(0) == 0:
            self.forward_output_tilde = forward_output_tilde
        else:
            if not forward_output_tilde.size(-2) == self.in_dim - \
                   self.layer_dim:
                raise ValueError(
                    "Expecting same dimension as in_dim - layer_dim")
            if not forward_output_tilde.size(-1) == 1:
                raise ValueError("Expecting same dimension as layer_dim")
            self.forward_output_tilde = forward_output_tilde

    def set_weight_update_u(self, u):
        """
        Save the u vector of the forward weight update to
         be able to use the Sherman-morrison formula
        ( https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula )
        """
        if not isinstance(u, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.u")
        self.u = u

    def set_weight_update_v(self, v):
        """
        Save the v vector of the forward weight update to
         be able to use the Sherman-morrison formula
        ( https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula )
        """
        if not isinstance(v, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.u")
        self.v = v

    def propagate_forward_tilde(self, lower_layer):
        """ Compute random features of the last layer activation in this
        layer, in order to make exact inverse computation possible for the
        backward pass"""

        if not isinstance(lower_layer, InvertibleLayer):
            raise TypeError("Expecting an InvertibleLayer object as "
                            "argument for propagate_forward_tilde")
        if not lower_layer.layer_dim == self.in_dim:
            raise ValueError("Layer sizes are not compatible for "
                             "propagating forward")

        forward_input = lower_layer.forward_output
        if self.forward_weights_tilde.size(0) > 0:
            linear_activation_tilde = torch.matmul(self.forward_weights_tilde,
                                                 forward_input) + \
                                    self.forward_bias_tilde
            forward_output_tilde = linear_activation_tilde
            # no need to put through
            # nonlinearity, as in the backward pass, the inverse non-linearity
            # would be applied. Now we skip both to save computation
        else:
            forward_output_tilde = torch.empty(0)
        self.set_forward_output_tilde(forward_output_tilde)

    def propagate_forward(self, lower_layer):
        """ Propagate the forward output as wel as the random features (
        forward_output_tilde."""
        super().propagate_forward(lower_layer)
        self.propagate_forward_tilde(lower_layer)

    def inverse_nonlinearity(self, input):
        """ Returns the inverse of the forward nonlinear activation function,
        performed on the given input.
        IMPORTANT: this function should always be overwritten by a child of
        InvertibleLayer, as now the forward nonlinearity is not yet specified"""
        raise NetworkError("inverse_nonlinearity should be overwritten by a "
                           "child of InvertibleLayer")

    def update_backward_parameters(self, learning_rate, upper_layer):
        """
        Update the backward weights and bias of the layer to resp.
        the inverse of the forward weights of the upper_layer and
        the negative bias of the upper_layer using the sherman-morrison
        formula
        """
        # take learning rate into u to apply Sherman-morrison formula later on
        u = torch.mul(upper_layer.u, -learning_rate)
        v = upper_layer.v
        if u.shape[0] < v.shape[0]:
            u = torch.cat((u, torch.zeros((v.shape[0] - u.shape[0],
                                           u.shape[1]))), 0)
        # apply Sherman-morrison formula to compute inverse

        d = torch.matmul(torch.transpose(v, -1, -2),
                         torch.matmul(self.backward_weights, u))
        self.d = d
        denominator = 1 + d
        self.denominator = denominator
        numerator = torch.matmul(torch.matmul(self.backward_weights, u),
                                 torch.matmul(torch.transpose(v, -1, -2),
                                              self.backward_weights))

        # Clipping for robustness, and adjusting forward weights to keep the
        # exact invertibility of sherman-morrison (see thesis chapter 4 for
        # the details
        epsilon = self.epsilon # threshold

        if torch.abs(denominator) < epsilon:
            self.beta = 1/(epsilon-d)
            backward_weights = self.backward_weights - torch.div(numerator,
                                                                 epsilon)
            # forward weights were already updated, so new update with (beta-1)
            # instead of beta
            forward_weights = upper_layer.forward_weights - \
                              upper_layer.forward_learning_rate*\
                              (self.beta - 1)*upper_layer.forward_weights_grad
            upper_layer.set_forward_parameters(forward_weights,
                                               upper_layer.forward_bias)
        else:
            backward_weights = self.backward_weights - torch.div(numerator,
                                                            denominator)
            self.beta = 1.

        backward_bias = - torch.cat((upper_layer.forward_bias,
                                    upper_layer.forward_bias_tilde), 0)
        self.set_backward_parameters(backward_weights, backward_bias)

    def propagate_backward(self, upper_layer):
        """Propagate the target signal from the upper layer to the current
        layer (self)
        :type upper_layer: InvertibleLayer
        """
        if not isinstance(upper_layer, InvertibleLayer):
            raise TypeError("Expecting an InvertibleLayer object as argument "
                            "for "
                            "propagate_backward")
        if not upper_layer.in_dim == self.layer_dim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")

        target_inverse = upper_layer.inverse_nonlinearity(
            upper_layer.backward_output)

        target_bar_inverse = torch.cat((target_inverse,
                                       upper_layer.forward_output_tilde), -2)

        backward_output = torch.matmul(self.backward_weights,
                                      target_bar_inverse + self.backward_bias)
        self.set_backward_output(backward_output)

    def compute_forward_gradients(self, lower_layer):
        """
        :param lower_layer: first layer upstream of the layer self
        :type lower_layer: Layer
        :return: saves the gradients of the local cost function to the layer
        parameters for all the batch samples

        """
        if lower_layer.forward_output.shape[0] > 1:
            raise NetworkError('only batch sizes of size 1 are allowed,'
                               ' as otherwise the '
                               'inverse computation with sherman-morrisson'
                               ' is not possible')
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
        self.set_weight_update_u(torch.reshape(u, (u.shape[-2], u.shape[-1])))
        self.set_weight_update_v(torch.reshape(v, (v.shape[-2], v.shape[-1])))
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
        forward_weights_bar = torch.cat((upper_layer.forward_weights,
                                       upper_layer.forward_weights_tilde), 0)
        error = torch.matmul(self.backward_weights, forward_weights_bar) \
                - torch.eye(self.backward_weights.shape[0])
        return torch.norm(error)

    def save_inverse_error(self, upper_layer):
        error = self.check_inverse(upper_layer)
        self.writer.add_scalar(tag='{}/inverse_error'.format(self.name),
                               scalar_value=error,
                               global_step=self.global_step)

    def save_sherman_morrison(self):
        self.writer.add_scalar(tag='{}/SM_beta'.format(self.name),
                               scalar_value=self.beta,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/SM_denominator'.format(self.name),
                               scalar_value=self.denominator,
                               global_step=self.global_step)


class InvertibleLeakyReluLayer(InvertibleLayer):
    """ Layer of an invertible neural network with a leaky RELU activation
    fucntion. """

    def __init__(self, negative_slope, in_dim, layer_dim,
                 out_dim, writer,
                 loss_function='mse',
                 name='invertible_leaky_ReLU_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 fixed=False):
        super().__init__(in_dim, layer_dim, out_dim,
                         writer=writer,
                         loss_function=loss_function,
                         name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
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


class InvertibleLinearLayer(InvertibleLayer):
    """ Layer in neural network that is purely linear and invertible"""

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def inverse_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)


class InvertibleOutputLayer(InvertibleLayer):
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

    def save_state_histograms(self):
        """ The histograms (specified by the arguments) are saved to
                tensorboard"""
        # Save histograms
        self.save_forward_weights_gradients_hist()
        self.save_forward_weights_hist()
        self.save_activations_hist()
        self.save_backward_activations_hist()


class InvertibleLinearOutputLayer(InvertibleOutputLayer):
    """ Invertible output layer with a linear activation function. This layer
    can so far only be combined with an mse loss
    function."""

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def inverse_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)

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


class InvertibleSoftmaxOutputLayer(InvertibleOutputLayer):
    """ Invertible output layer with a linear activation function. This layer
    can so far only be combined with an mse loss
    function."""

    def __init__(self, in_dim, layer_dim, writer, step_size,
                 loss_function='mse',
                 output_loss_function='crossEntropy',
                 name='invertible_softmax_output_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 fixed=False):
        super().__init__(in_dim, layer_dim,
                         writer=writer,
                         step_size=step_size,
                         loss_function=loss_function,
                         output_loss_function=output_loss_function,
                         name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)
        self.normalization_constant = None

    def forward_nonlinearity(self, linear_activation):
        self.normalization_constant = torch.logsumexp(linear_activation, 1)
        softmax = nn.Softmax(dim=1)
        # print('original linear activation: {}'.format(linear_activation))
        return softmax(linear_activation)

    def inverse_nonlinearity(self, input):
        # print('computed linear activation: {}'.format(
        #     torch.log(input) + self.normalization_constant))
        return torch.log(input) + self.normalization_constant

    def compute_vectorized_jacobian(self):
        raise NotImplementedError('Softmax outputlayer has a custom '
                                  'implementation of compute_forward_gradients'
                                  'without the usage of '
                                  'compute_vectorized_jacobian')

    def compute_backward_output(self, target):
        """ Compute the backward output based on a small move from the
        forward output in the direction of the negative gradient of the loss
        function."""
        if not self.output_loss_function == 'crossEntropy':
            raise NetworkError("a softmax output layer can only be combined "
                               "with a crossEntropy loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension '
                             'x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        gradient = self.forward_output - target
        self.backward_output = self.forward_output - torch.mul(gradient,
                                                               self.step_size)

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
        if not self.forward_output.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension'
                             ' x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        return hf.accuracy(self.predicted_classes, hf.prob2class(target))

    def compute_forward_gradients(self, lower_layer):
        """
        :param lower_layer: first layer upstream of the layer self
        :type lower_layer: Layer
        :return: saves the gradients of the local cost function to the layer
        parameters for all the batch samples

        """
        if lower_layer.forward_output.shape[0] > 1:
            raise NetworkError('only batch sizes of size 1 are allowed,'
                               ' as otherwise the '
                               'inverse computation with sherman-morrisson'
                               ' is not possible')
        if not self.loss_function == 'mse':
            raise NetworkError('only mse loss function is defined as a '
                               'local layer costfuntion. Now got'
                               ' {}'.format(self.loss_function))

        u = self.forward_output - self.backward_output
        v = lower_layer.forward_output

        weight_gradients = torch.matmul(u, torch.transpose(v, -1, -2))

        # bias_gradients = u
        bias_gradients = torch.zeros(u.shape)
        self.set_weight_update_u(torch.reshape(u, (u.shape[-2], u.shape[-1])))
        self.set_weight_update_v(torch.reshape(v, (v.shape[-2], v.shape[-1])))
        self.set_forward_gradients(torch.mean(weight_gradients, 0), torch.mean(
            bias_gradients, 0))


class InvertibleInputLayer(InvertibleLayer):
    """ Input layer of the invertible neural network,
        e.g. the pixelvalues of a picture. """

    def __init__(self, layer_dim, out_dim, writer,
                 loss_function='mse',
                 name='invertible_input_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 fixed=False):
        super().__init__(in_dim=None, layer_dim=layer_dim,
                         out_dim=out_dim,
                         writer=writer,
                         loss_function=loss_function,
                         name=name,
                         debug_mode=debug_mode,
                         weight_decay=weight_decay,
                         fixed=fixed)

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


class InvertibleClassificationOutputLayer(InvertibleOutputLayer):
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
        if not isinstance(self, InvertibleCapsuleOutputLayer):
            if not self.forward_output.shape == target.shape:
                raise ValueError(
                    'Expecting a tensor of dimensions: batchdimension'
                    ' x class dimension x 1. Given target'
                    'has shape' + str(target.shape))
        return hf.accuracy(self.predicted_classes, hf.prob2class(target))

class InvertibleCapsuleOutputLayer(InvertibleClassificationOutputLayer):
    """ Output layer with capsule loss for an invertible network. """
    def __init__(self, in_dim, layer_dim, nb_classes, writer, step_size,
                 output_loss_function='capsule_loss',
                 loss_function='mse',
                 name='invertible_capsule_output_layer',
                 debug_mode=True,
                 weight_decay=0.0,
                 fixed=False):
        super().__init__(in_dim, layer_dim,
                         writer=writer,
                         step_size=step_size,
                         loss_function=loss_function,
                         output_loss_function=output_loss_function,
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
        super(InvertibleClassificationOutputLayer, self).propagate_forward(
            lower_layer)
        self.compute_capsules()
        self.predicted_classes = hf.prob2class(self.capsule_squashed)

    def forward_nonlinearity(self, linear_activation):
        return linear_activation

    def inverse_nonlinearity(self, input):
        return input

    def compute_vectorized_jacobian(self):
        return torch.ones(self.forward_output.shape)

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
                                     self.capsule_indices[k][0]:
                                     self.capsule_indices[k][1], 0]
            else:
                self.capsules[:, k, 0:self.capsule_base_size] = \
                    linear_activation[:, self.capsule_indices[k][0]:
                                      self.capsule_indices[k][1], 0]
            self.capsule_magnitudes[:, k, 0] = torch.norm(
                self.capsules[:, k, :], dim=1)

        self.capsule_squashed = self.capsule_magnitudes ** 2 / (
                1 + self.capsule_magnitudes ** 2)

    def loss(self, target):
        if self.output_loss_function == 'capsule_loss':
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
                self.output_loss_function))

    def compute_backward_output(self, target):
        """ Compute the backward output based on the derivative of the loss to
        the linear activation of this layer"""
        if not self.output_loss_function == 'capsule_loss':
            raise NetworkError("Only capsule_loss is defined for a capsule"
                               "output layer, got {}".format(
                self.output_loss_function))
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
                                                    0:self.capsule_base_size]*\
                vk_sk[:, k, 0:self.capsule_base_size]
        backward_output = self.forward_output - torch.mul(backward_output,
                                                               self.step_size)
        self.set_backward_output(backward_output)

    def init_forward_parameters(self):
        """ Initializes the layer parameters when the layer is created.
                This method should only be used when creating
                a new layer. Use set_forward_parameters to update the parameters and
                computeGradient to update the gradients"""
        super().init_forward_parameters()
        # self.forward_weights = self.forward_weights / float(self.layer_dim)**0.5
