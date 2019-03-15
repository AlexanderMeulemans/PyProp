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
from utils import HelperFunctions as hf
from utils.HelperClasses import NetworkError
from tensorboardX import SummaryWriter


class Layer(object):
    """ Parent class of all occurring layers in neural networks with only
    feedforward weights. This class should not be used directly,
    only via its children"""
    # create class variable of existing layer names
    allLayerNames = []

    def __init__(self, inDim, layerDim, writer, name='layer'):
        """
        Initializes the Layer object
        :param inDim: input dimension of the layer (equal to the layer dimension
        of the previous layer in the network)
        :param layerDim: Layer dimension
        """
        self.setLayerDim(layerDim)
        if inDim is not None: # inDim is None when layer is inputlayer
            self.setInDim(inDim)
        self.setName(name)
        self.setWriter(writer=writer)
        self.initForwardParameters()
        self.global_step = 0 # needed for making plots with tensorboard

    def setWriter(self, writer):
        if not isinstance(writer,SummaryWriter):
            raise TypeError("Writer object has to be of type "
                            "SummaryWriter, now got {}".format(
                type(writer)))
        self.writer = writer

    def setLayerDim(self, layerDim):
        if not isinstance(layerDim, int):
            raise TypeError("Expecting an integer layer dimension")
        if layerDim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.layerDim = layerDim

    def setInDim(self, inDim):
        if not isinstance(inDim, int):
            raise TypeError("Expecting an integer layer dimension")
        if inDim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.inDim = inDim

    def setName(self, name):
        if not isinstance(name, str):
            raise TypeError("Expecting a string as name for the layer")
        if not name in self.__class__.allLayerNames:
            self.name = name
            self.__class__.allLayerNames.append(name)
        else:
            newName = name
            i=1
            while newName in self.__class__.allLayerNames:
                newName = name + '_' + str(i)
                i += 1
            self.name = newName
            self.__class__.allLayerNames.append(name)

    def setForwardParameters(self, forwardWeights, forwardBias):
        if not isinstance(forwardWeights, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forwardWeights")
        if not isinstance(forwardBias, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forwardBias")
        if hf.containsNaNs(forwardWeights):
            raise ValueError("forwardWeights contains NaNs")
        if hf.containsNaNs(forwardBias):
            raise ValueError("forwardBias contains NaNs")
        if not forwardWeights.shape == self.forwardWeights.shape:
            raise ValueError("forwardWeights has not the correct shape")
        if not forwardBias.shape == self.forwardBias.shape:
            raise ValueError("forwardBias has not the correct shape")

        if torch.max(torch.abs(forwardWeights))> 1e3:
            print('forwardWeights of {} have gone unbounded'.format(
                self.name))
        if torch.max(torch.abs(forwardBias)) > 1e3:
            print('forwardBiases of {} have gone unbounded'.format(
                self.name))
        self.forwardWeights = forwardWeights
        self.forwardBias = forwardBias

    def setForwardGradients(self, forwardWeightsGrad, forwardBiasGrad):
        if not isinstance(forwardWeightsGrad, torch.Tensor):
            raise TypeError("Expecting a tensor object "
                            "for self.forwardWeightsGrad")
        if not isinstance(forwardBiasGrad, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forwardBiasGrad")
        if hf.containsNaNs(forwardWeightsGrad):
            raise ValueError("forwardWeightsGrad contains NaNs")
        if hf.containsNaNs(forwardBiasGrad):
            raise ValueError("forwardBias contains NaNs")
        if not forwardWeightsGrad.shape == self.forwardWeightsGrad.shape:
            raise ValueError("forwardWeightsGrad has not the correct shape")
        if not forwardBiasGrad.shape == self.forwardBiasGrad.shape:
            raise ValueError("forwardBiasGrad has not the correct shape")

        if torch.max(torch.abs(forwardWeightsGrad))> 1e3:
            print('forwardWeightsGrad of {} have gone unbounded'.format(
                self.name))
        if torch.max(torch.abs(forwardBiasGrad)) > 1e3:
            print('forwardBiasesGrad of {} have gone unbounded'.format(
                self.name))
        self.forwardWeightsGrad = forwardWeightsGrad
        self.forwardBiasGrad = forwardBiasGrad

    def setForwardOutput(self, forwardOutput):
        if not isinstance(forwardOutput, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forwardOutput")
        if not forwardOutput.size(-2) == self.layerDim:
            raise ValueError("Expecting same dimension as layerDim")
        if not forwardOutput.size(-1) == 1:
            raise ValueError("Expecting same dimension as layerDim")
        self.forwardOutput = forwardOutput

    def setForwardLinearActivation(self, forwardLinearActivation):
        if not isinstance(forwardLinearActivation, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forwardLinearActivation")
        if not forwardLinearActivation.size(-2) == self.layerDim:
            raise ValueError("Expecting same dimension as layerDim")
        if not forwardLinearActivation.size(-1) == 1:
            raise ValueError("Expecting same dimension as layerDim")
        self.forwardLinearActivation = forwardLinearActivation

    def setBackwardOutput(self, backwardOutput):
        if not isinstance(backwardOutput, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.backwardOutput")
        if not backwardOutput.size(-2) == self.layerDim:
            raise ValueError("Expecting same dimension as layerDim")
        if not backwardOutput.size(-1) == 1:
            raise ValueError("Expecting same dimension as layerDim")
        self.backwardOutput = backwardOutput
        if torch.max(torch.abs(backwardOutput))> 1e2:
            print('backwardOutputs of {} have gone unbounded: {}'.format(
                self.name, torch.max(torch.abs(backwardOutput))))


    def initForwardParameters(self):
        """ Initializes the layer parameters when the layer is created.
        This method should only be used when creating
        a new layer. Use setForwardParameters to update the parameters and
        computeGradient to update the gradients"""
        self.forwardWeights = torch.randn(self.layerDim, self.inDim)
        self.forwardBias = torch.zeros(self.layerDim, 1)
        self.forwardWeightsGrad = torch.zeros(self.layerDim, self.inDim)
        self.forwardBiasGrad = torch.zeros(self.layerDim, 1)
        self.save_initial_state()

    def initVelocities(self):
        """ Initializes the velocities of the gradients. This should only be
        called when an optimizer with momentum
        is used, otherwise these attributes will not be used"""
        self.forwardWeightsVel = torch.zeros(self.layerDim, self.inDim)
        self.forwardBiasVel = torch.zeros(self.layerDim, 1)

    def setForwardVelocities(self, forwardWeightsVel, forwardBiasVel):
        if not isinstance(forwardWeightsVel, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forwardWeightsVel")
        if not isinstance(forwardBiasVel, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forwardBiasVel")
        if hf.containsNaNs(forwardWeightsVel):
            raise ValueError("forwardWeightsVel contains NaNs")
        if hf.containsNaNs(forwardBiasVel):
            raise ValueError("forwardBiasVel contains NaNs")
        if not forwardWeightsVel.shape == self.forwardWeightsVel.shape:
            raise ValueError("forwardWeightsVel has not the correct shape")
        if not forwardBiasVel.shape == self.forwardBiasVel.shape:
            raise ValueError("forwardBiasVel has not the correct shape")

        self.forwardWeightsVel = forwardWeightsVel
        self.forwardBiasVel = forwardBiasVel

    def zeroGrad(self):
        """ Set the gradients of the layer parameters to zero """
        self.forwardWeightsGrad = torch.zeros(self.layerDim, self.inDim)
        self.forwardBiasGrad = torch.zeros(self.layerDim, 1)

    def updateForwardParameters(self, learningRate):
        """
        Update the forward weights and bias of the layer
        using the computed gradients.
        :param learningRate: Learning rate of the layer
        """
        if not isinstance(learningRate, float):
            raise TypeError("Expecting a float number as learningRate")
        if learningRate <= 0.:
            raise ValueError("Expecting a strictly positive learningRate")

        forwardWeights = self.forwardWeights \
                         - torch.mul(self.forwardWeightsGrad, learningRate)
        forwardBias = self.forwardBias \
                      - torch.mul(self.forwardBiasGrad, learningRate)
        self.setForwardParameters(forwardWeights, forwardBias)

    def propagateForward(self, lowerLayer):
        """
        :param lowerLayer: The first layer upstream of the layer 'self'
        :type lowerLayer: Layer
        :return saves the computed output of the layer to self.forwardOutput.
                forwardOutput is a 3D tensor of size
                batchDimension x layerDimension x 1
        """
        if not isinstance(lowerLayer, Layer):
            raise TypeError("Expecting a Layer object as "
                            "argument for propagateForward")
        if not lowerLayer.layerDim == self.inDim:
            raise ValueError("Layer sizes are not compatible for "
                             "propagating forward")

        self.forwardInput = lowerLayer.forwardOutput
        forwardLinearActivation = torch.matmul(self.forwardWeights,
                                             self.forwardInput) + \
                                self.forwardBias
        self.setForwardLinearActivation(forwardLinearActivation)
        forwardOutput = self.forwardNonlinearity(self.forwardLinearActivation)
        self.setForwardOutput(forwardOutput)
        if torch.max(torch.abs(forwardOutput))> 1e3:
            print('forwardOutputs of {} have gone unbounded'.format(
                self.name))

    def forwardNonlinearity(self, linearActivation):
        """ This method should be always overwritten by the children"""
        raise NetworkError("The method forwardNonlinearity should always be "
                           "overwritten by children of Layer. Layer on itself "
                           "cannot be used in a network")

    def computeForwardGradients(self, lowerLayer):
        """
        :param lowerLayer: first layer upstream of the layer self
        :type lowerLayer: Layer
        :return: saves the gradients of the cost function to the layer
        parameters for all the batch samples

        """

        weight_gradients = torch.matmul(self.backwardOutput, torch.transpose(
            lowerLayer.forwardOutput, -1, -2))
        bias_gradients = self.backwardOutput
        self.setForwardGradients(torch.mean(weight_gradients, 0), torch.mean(
            bias_gradients, 0))

    def computeForwardGradientVelocities(self, lowerLayer, momentum,
                                       learningRate):
        """ Used for optimizers with momentum. """
        if not isinstance(momentum, float):
            raise TypeError("Expecting float number for momentum, "
                            "got {}".format(type(momentum)))
        if not (momentum >= 0. and momentum < 1.):
            raise ValueError("Expecting momentum in [0;1), got {}".format(
                momentum))

        # weight_gradients = torch.mean(torch.matmul(self.backwardOutput,
        #                                            torch.transpose(
        #                                              lowerLayer.forwardOutput,
        #                                                -1, -2)), 0)
        # bias_gradients = torch.mean(self.backwardOutput, 0)
        weight_gradients = self.forwardWeightsGrad
        bias_gradients = self.forwardBiasGrad
        weight_velocities = torch.mul(self.forwardWeightsVel, momentum) + \
                            torch.mul(weight_gradients, learningRate)
        bias_velocities = torch.mul(self.forwardBiasVel, momentum) + \
                          torch.mul(bias_gradients, learningRate)
        self.setForwardVelocities(weight_velocities, bias_velocities)

    def updateForwardParametersWithVelocity(self):
        """ Update the forward parameters with the gradient velocities
        computed in computeForwardGradientVelocities"""
        forwardWeights = self.forwardWeights \
                         - self.forwardWeightsVel
        forwardBias = self.forwardBias \
                      - self.forwardBiasVel
        self.setForwardParameters(forwardWeights, forwardBias)

    def propagateBackward(self, upperLayer):
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
        activations_norm = torch.norm(self.forwardOutput)
        self.writer.add_scalar(tag='{}/forward_activations'
                                   '_norm'.format(self.name),
                               scalar_value=activations_norm,
                               global_step=self.global_step)

    def save_activations_hist(self):
        self.writer.add_histogram(tag='{}/forward_activations_'
                                      'hist'.format(
            self.name),
            values=self.forwardOutput,
            global_step=self.global_step)

    def save_forward_weights(self):
        weight_norm = torch.norm(self.forwardWeights)
        bias_norm = torch.norm(self.forwardBias)
        self.writer.add_scalar(tag='{}/forward_weights'
                                   '_norm'.format(self.name),
                               scalar_value=weight_norm,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/forward_bias'
                                   '_norm'.format(self.name),
                               scalar_value=bias_norm,
                               global_step=self.global_step)

    def save_forward_weight_gradients(self):
        gradient_norm = torch.norm(self.forwardWeightsGrad)
        gradient_bias_norm = torch.norm(self.forwardBiasGrad)

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
            values=self.forwardWeightsGrad,
            global_step=self.global_step
        )
        self.writer.add_histogram(tag='{}/forward_bias_'
                                      'gradient_hist'.format(
            self.name),
            values=self.forwardBiasGrad,
            global_step=self.global_step
        )

    def save_forward_weights_hist(self):
        self.writer.add_histogram(tag='{}/forward_weights_'
                                      'hist'.format(self.name),
            values=self.forwardWeights,
            global_step=self.global_step)
        self.writer.add_histogram(tag='{}/forward_bias_'
                                      'hist'.format(
            self.name),
            values=self.forwardBias,
            global_step=self.global_step)

    def save_initial_state(self):
        self.writer.add_histogram(tag='{}/forward_weights_initial_'
                                      'hist'.format(
            self.name),
            values=self.forwardWeights,
            global_step=0)
        self.writer.add_histogram(tag='{}/forward_bias_initial_'
                                      'hist'.format(
            self.name),
            values=self.forwardBias,
            global_step=0)



class ReluLayer(Layer):
    """ Layer of a neural network with a RELU activation function"""

    def forwardNonlinearity(self, linearActivation):
        """ Returns the nonlinear activation of the layer"""
        return F.relu(linearActivation)

    def propagateBackward(self, upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backwardOutput is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upperLayer, Layer):
            raise TypeError("Expecting a Layer object as argument for "
                            "propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")

        self.backwardInput = upperLayer.backwardOutput
        # Construct vectorized Jacobian for all batch samples.
        activationDer = torch.tensor(
            [[[1.] if self.forwardLinearActivation[i, j, 0] > 0
              else [0.]
              for j in range(self.forwardLinearActivation.size(1))]
             for i in range(self.forwardLinearActivation.size(0))])
        backwardOutput = torch.mul(torch.matmul(torch.transpose(
            upperLayer.forwardWeights, -1, -2), self.backwardInput),
            activationDer)
        self.setBackwardOutput(backwardOutput)
        if torch.max(torch.abs(backwardOutput))> 1e3:
            print('backwardOutputs of {} have gone unbounded'.format(
                self.name))
            print('max backwardInput: {}'.format(torch.max(torch.abs(
                self.backwardInput))))
            print('upper layer max forward weights: {}'.format(
                torch.max(torch.abs(upperLayer.forwardWeights))
            ))
            # print('Jacobian:')
            # print(activationDer)

class LeakyReluLayer(Layer):
    """ Layer of a neural network with a Leaky RELU activation function"""
    def __init__(self,negativeSlope, inDim, layerDim, writer,
                 name='leaky_ReLU_layer'):
        super().__init__(inDim, layerDim, writer, name=name)
        self.setNegativeSlope(negativeSlope)

    def setNegativeSlope(self, negativeSlope):
        """ Set the negative slope of the leaky ReLU activation function"""
        if not isinstance(negativeSlope,float):
            raise TypeError("Expecting a float number for negativeSlope, "
                            "got {}".format(type(negativeSlope)))
        if negativeSlope <= 0:
            raise ValueError("Expecting a strictly positive float number for "
                             "negativeSlope, got {}".format(negativeSlope))

        self.negativeSlope = negativeSlope

    def forwardNonlinearity(self, linearActivation):
        activationFunction = nn.LeakyReLU(self.negativeSlope)
        return activationFunction(linearActivation)


    def propagateBackward(self, upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backwardOutput is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upperLayer, Layer):
            raise TypeError("Expecting a Layer object as argument for "
                            "propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")

        self.backwardInput = upperLayer.backwardOutput
        # Construct vectorized Jacobian for all batch samples.
        activationDer = torch.tensor(
            [[[1.] if self.forwardLinearActivation[i, j, 0] > 0
              else [self.negativeSlope]
              for j in range(self.forwardLinearActivation.size(1))]
             for i in range(self.forwardLinearActivation.size(0))])
        backwardOutput = torch.mul(torch.matmul(torch.transpose(
            upperLayer.forwardWeights, -1, -2), self.backwardInput),
            activationDer)
        self.setBackwardOutput(backwardOutput)

class SoftmaxLayer(Layer):
    """ Layer of a neural network with a Softmax activation function"""

    def forwardNonlinearity(self, linearActivation):
        """ Returns the nonlinear activation of the layer"""
        softmax = torch.nn.Softmax(1)
        return softmax(linearActivation)

    def propagateBackward(self, upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backwardOutput is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upperLayer, Layer):
            raise TypeError("Expecting a Layer object as argument for  "
                            "propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible for "
                             "propagating backwards")

        self.backwardInput = upperLayer.backwardOutput
        # Construct Jacobian for all batch samples.
        softmaxActivations = self.forwardOutput
        jacobian = torch.tensor([[[softmaxActivations[i, j, 0] *
                                   (hf.kronecker(j, k) - softmaxActivations[
                                       i, k, 0])
                                   for k in range(softmaxActivations.size(1))]
                                  for j in range(softmaxActivations.size(1))]
                                 for i in range(softmaxActivations.size(0))])
        backwardOutput = torch.matmul(torch.transpose(jacobian, -1, -2),
                                      torch.matmul(torch.transpose(
                                          upperLayer.forwardWeights, -1, -2)
                                          , self.backwardInput))
        self.setBackwardOutput(backwardOutput)


class LinearLayer(Layer):
    """ Layer of a neural network with a linear activation function"""

    def forwardNonlinearity(self, linearActivation):
        """ Returns the nonlinear activation of the layer"""
        return linearActivation

    def propagateBackward(self, upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backwardOutput is of
        size batchDimension x layerdimension  x 1
        """
        if not isinstance(upperLayer, Layer):
            raise TypeError("Expecting a Layer object as "
                            "argument for propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible "
                             "for propagating backwards")

        self.backwardInput = upperLayer.backwardOutput
        backwardOutput = torch.matmul(torch.transpose(
            upperLayer.forwardWeights, -1, -2), self.backwardInput)
        self.setBackwardOutput(backwardOutput)


class InputLayer(Layer):
    """ Input layer of the neural network,
    e.g. the pixelvalues of a picture. """

    def __init__(self, layerDim, writer, name='input_layer'):
        """ InputLayer has only a layerDim and a
        forward activation that can be set,
         no input dimension nor parameters"""
        super().__init__(inDim = None, layerDim=layerDim, writer=writer,
                         name=name)

    def propagateForward(self, lowerLayer):
        """ This function should never be called for an input layer,
        the forwardOutput should be directly set
        to the input values of the network (e.g. the pixel values of a picture)
        """
        raise NetworkError("The forwardOutput should be directly set "
                           "to the input values of the network for "
                           "an InputLayer")

    def propagateBackward(self, upperLayer):
        """ This function should never be called for an input layer,
        there is no point in having a backward output
         here, as this layer has no parameters to update"""
        raise NetworkError("Propagate Backward should never be called for "
                           "an input layer, there is no point in having "
                           "a backward output here, as this layer has no "
                           "parameters to update")
    def initForwardParameters(self):
        """ InputLayer has no forward parameters"""
        pass

    def initVelocities(self):
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

    def __init__(self, inDim, layerDim, lossFunction, writer,
                 name='output_layer'):
        """
        :param inDim: input dimension of the layer,
        equal to the layer dimension of the second last layer in the network
        :param layerDim: Layer dimension
        :param loss: string indicating which loss function is used to
        compute the network loss
        """
        super().__init__(inDim, layerDim, writer, name=name)
        self.setLossFunction(lossFunction)

    def setLossFunction(self, lossFunction):
        if not isinstance(lossFunction, str):
            raise TypeError('Expecting a string as indicator'
                            ' for the loss function')
        if not (lossFunction == 'mse' or lossFunction == 'crossEntropy'):
            raise NetworkError('Expecting an mse or crossEntropy loss')
        self.lossFunction = lossFunction

    def loss(self, target):
        """ Compute the loss with respect to the target
        :param target: 3D tensor of size batchdimension x class dimension x 1
        """
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension'
                             ' x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        if self.lossFunction == 'crossEntropy':
            # Convert output 'class probabilities' to one class per batch sample
            #  (with highest class probability)
            target_classes = hf.prob2class(target)
            lossFunction = nn.CrossEntropyLoss()
            forwardOutputSqueezed = torch.reshape(self.forwardOutput,
                                                  (self.forwardOutput.shape[0],
                                                   self.forwardOutput.shape[1]))
            loss = lossFunction(forwardOutputSqueezed, target_classes)
            return torch.Tensor([torch.mean(loss)])#.numpy()
        elif self.lossFunction == 'mse':
            lossFunction = nn.MSELoss(reduction='mean')
            forwardOutputSqueezed = torch.reshape(self.forwardOutput,
                                                  (self.forwardOutput.shape[0],
                                                   self.forwardOutput.shape[1]))
            targetSqueezed = torch.reshape(target,
                                            (target.shape[0],
                                             target.shape[1]))
            loss = lossFunction(forwardOutputSqueezed, targetSqueezed)
            loss = torch.Tensor([torch.mean(loss)])
            if loss > 1e2:
                print('loss is bigger than 100. Loss: {}'.format(loss))
                print('max difference: {}'.format(torch.max(torch.abs(
                    forwardOutputSqueezed-targetSqueezed
                ))))

            return loss

    def propagateBackward(self, upperLayer):
        """ This function should never be called for an output layer,
        the backwardOutput should be set based on the
        loss of the layer with computeBackwardOutput"""
        raise NetworkError("Propagate Backward should never be called for an "
                           "output layer, use computeBackwardOutput "
                           "instead")


class SoftmaxOutputLayer(OutputLayer):
    """ Output layer with a softmax activation function. This layer
    should always be combined with a crossEntropy
    loss."""

    def forwardNonlinearity(self, linearActivation):
        """ Returns the nonlinear activation of the layer"""
        softmax = torch.nn.Softmax(1)
        return softmax(linearActivation)

    def computeBackwardOutput(self, target):
        """ Compute the backward output based on the derivative of the loss
        to the linear activation of this layer
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        if not self.lossFunction == 'crossEntropy':
            raise NetworkError("a softmax output layer should always be "
                               "combined with a cross entropy loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: '
                             'batchdimension x class dimension x 1.'
                             ' Given target'
                             'has shape' + str(target.shape))

        backwardOutput = self.forwardOutput - target
        self.setBackwardOutput(backwardOutput)

    def propagateForward(self, lowerLayer):
        """ Normal forward propagation, but on top of that, save the predicted
        classes in self."""
        super().propagateForward(lowerLayer)
        self.predictedClasses = hf.prob2class(self.forwardOutput)

    def accuracy(self, target):
        """ Compute the accuracy if the network predictions with respect to
        the given true targets.
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension'
                             ' x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        return hf.accuracy(self.predictedClasses, hf.prob2class(target))


class LinearOutputLayer(OutputLayer):
    """ Output layer with a linear activation function. This layer can so far
    only be combined with an mse loss
    function."""

    def forwardNonlinearity(self, linearActivation):
        """ Returns the nonlinear activation of the layer"""
        return linearActivation

    def computeBackwardOutput(self, target):
        """ Compute the backward output based on the derivative of the loss to
        the linear activation of this layer"""
        if not self.lossFunction == 'mse':
            raise NetworkError("a linear output layer can only be combined "
                               "with a mse loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension '
                             'x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        backwardOutput = 2 * (self.forwardOutput - target)
        self.setBackwardOutput(backwardOutput)
        if torch.max(torch.abs(backwardOutput)) > 1e2:
            print('backwardOutputs of {} have gone unbounded: {}'.format(
            self.name, torch.max(torch.abs(backwardOutput))))



