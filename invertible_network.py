import torch
import torch.nn as nn
import torch.nn.functional as F
import HelperFunctions as hf
from HelperClasses import NetworkError
import numpy as np
from neuralnetwork import Layer, Network
from bidirectional_layer import BidirectionalLayer


class InvertibleLayer(BidirectionalLayer):
    """ Layer that is invertible to make it able to propagate exact targets."""

    def __init__(self,inDim, layerDim, outDim, lossFunction = 'mse'):
        if inDim is not None:
            if inDim < layerDim:
                raise ValueError("Expecting an input size bigger or equal to the "
                                 "layer size")
        if outDim is not None:
            if layerDim < outDim:
                raise ValueError("Expecting a layer size bigger or equal to the "
                                 "output size")
        super().__init__(inDim, layerDim, outDim, lossFunction)
        self.initForwardParametersTilde()

    def initForwardParametersTilde(self):
        """ Initializes the layer parameters that connect the current layer
        with the random fixed features of the next layer
        when the layer is created.
        This method should only be used when creating
        a new layer. These parameters should remain fixed"""
        self.forwardWeightsTilde = torch.rand(self.inDim - self.layerDim,
                                              self.inDim)
        self.forwardBiasTilde = torch.zeros(self.inDim - self.layerDim, 1)

    def initBackwardParameters(self):
        """ Initializes the layer parameters when the layer is created.
        This method should only be used when creating
        a new layer. Use setbackwardParameters to update the parameters and
        computeGradient to update the gradients"""
        self.backwardWeights = torch.empty(self.layerDim, self.layerDim)
        self.backwardBias = torch.empty(self.layerDim, 1)\

    # def initForwardParametersBar(self):
    #     """ Concatenates the forwardWeights with the forwardWeightsTilde to
    #     create a square matrix ForwardWeightsBar. Similarly concatenates the
    #     two biases. """
    #     self.forwardWeightsBar = torch.cat((self.forwardWeights,
    #                                         self.forwardWeightsTilde), 0)
    #     self.forwardBiasBar = torch.cat((self.forwardBias,
    #                                      self.forwardBiasTilde), 0)

    def setForwardOutputTilde(self, forwardOutputTilde):
        if not isinstance(forwardOutputTilde, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.forwardOutputTilde")
        if forwardOutputTilde.size(0) == 0:
            self.forwardOutputTilde = forwardOutputTilde
        else:
            if not forwardOutputTilde.size(-2) == self.inDim - self.layerDim:
                raise ValueError("Expecting same dimension as inDim - layerDim")
            if not forwardOutputTilde.size(-1) == 1:
                raise ValueError("Expecting same dimension as layerDim")
            self.forwardOutputTilde = forwardOutputTilde

    def propagateForwardTilde(self, lowerLayer):
        """ Compute random features of the last layer activation in this
        layer, in order to make exact inverse computation possible for the
        backward pass"""

        if not isinstance(lowerLayer, InvertibleLayer):
            raise TypeError("Expecting an InvertibleLayer object as "
                            "argument for propagateForwardTilde")
        if not lowerLayer.layerDim == self.inDim:
            raise ValueError("Layer sizes are not compatible for "
                             "propagating forward")

        forwardInput = lowerLayer.forwardOutput
        if self.forwardWeightsTilde.size(0)>0:
            linearActivationTilde = torch.matmul(self.forwardWeightsTilde,
                                                 forwardInput) + \
                                    self.forwardBiasTilde
            forwardOutputTilde = self.forwardNonlinearity(linearActivationTilde)
        else:
            forwardOutputTilde = torch.empty(0)
        self.setForwardOutputTilde(forwardOutputTilde)

    def propagateForward(self, lowerLayer):
        """ Propagate the forward output as wel as the random features (
        forwardOutputTilde."""
        super().propagateForward(lowerLayer)
        self.propagateForwardTilde(lowerLayer)

    def inverseNonlinearity(self,input):
        """ Returns the inverse of the forward nonlinear activation function,
        performed on the given input.
        IMPORTANT: this function should always be overwritten by a child of
        InvertibleLayer, as now the forward nonlinearity is not yet specified"""
        raise NetworkError("inverseNonlinearity should be overwritten by a "
                           "child of InvertibleLayer")

    def propagateBackward(self, upperLayer):
        """Propagate the target signal from the upper layer to the current
        layer (self)
        TODO(finish this function)
        :type upperLayer: InvertibleLayer
        """
        if not isinstance(upperLayer, InvertibleLayer):
            raise TypeError("Expecting an InvertibleLayer object as argument "
                            "for "
                            "propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible for propagating "
                             "backwards")

        backwardWeights = torch.inverse(torch.cat((upperLayer.forwardWeights,
                                            upperLayer.forwardWeightsTilde), 0))
        backwardBias = - torch.cat((upperLayer.forwardBias,
                                         upperLayer.forwardBiasTilde), 0)
        self.setBackwardParameters(backwardWeights, backwardBias)

        targetBar = torch.cat((upperLayer.backwardOutput,
                               upperLayer.forwardOutputTilde), -2)

        backwardOutput = torch.matmul(self.backwardWeights,
                                      (upperLayer.inverseNonlinearity(
                                          targetBar) + self.backwardBias))
        self.setBackwardOutput(backwardOutput)

    def computeForwardGradients(self, lowerLayer):
        """
        :param lowerLayer: first layer upstream of the layer self
        :type lowerLayer: Layer
        :return: saves the gradients of the local cost function to the layer
        parameters for all the batch samples

        """
        if self.lossFunction == 'mse':
            localLossDer = torch.mul(self.forwardOutput -
                                     self.backwardOutput, 2.)
        else:
            raise NetworkError("Expecting a mse local loss function")

        vectorizedJacobian = self.computeVectorizedJacobian()

        weight_gradients = torch.matmul(torch.mul(vectorizedJacobian,
                                                  localLossDer),
                                        torch.transpose(
                                            lowerLayer.forwardOutput, -1, -2))
        bias_gradients = torch.mul(vectorizedJacobian, localLossDer)
        self.setForwardGradients(torch.mean(weight_gradients, 0), torch.mean(
            bias_gradients, 0))

    def computeVectorizedJacobian(self):
        """ Compute the vectorized Jacobian (as the jacobian for a ridge
        nonlinearity is diagonal, it can be stored in a vector.
        IMPORTANT: this function should always be overwritten by children of
        InvertibleLayer"""
        raise NetworkError("computeVectorizedJacobian should always be "
                           "overwritten by children of InvertibleLayer")


class InvertibleLeakyReluLayer(InvertibleLayer):
    """ Layer of an invertible neural network with a leaky RELU activation
    fucntion. """

    def __init__(self,negativeSlope, inDim, layerDim, outDim, lossFunction =
        'mse'):
        super().__init__(inDim, layerDim, outDim, lossFunction)
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

    def inverseNonlinearity(self,input):
        """ perform the inverse of the forward nonlinearity on the given
        input. """
        output = torch.empty(input.shape)
        for i in range(input.size(0)):
            for j in range(input.size(1)):
                for k in range(input.size(2)):
                    if input[i,j,k] >= 0:
                        output[i,j,k] = input[i,j,k]
                    else:
                        output[i,j,k] = input[i,j,k]/self.negativeSlope
        return output

    def computeVectorizedJacobian(self):
        """ Compute the vectorized jacobian. The jacobian is a diagonal
        matrix, so can be represented by a vector instead of a matrix. """

        output = torch.empty(self.forwardOutput.shape)
        for i in range(self.forwardLinearActivation.size(0)):
            for j in range(self.forwardLinearActivation.size(1)):
                if self.forwardLinearActivation[i,j,0] >= 0:
                    output[i,j,0] = 1
                else:
                    output[i,j,0] = self.negativeSlope
        return output


class InvertibleLinearLayer(InvertibleLayer):
    """ Layer in neural network that is purely linear and invertible"""

    def forwardNonlinearity(self, linearActivation):
        return linearActivation

    def inverseNonlinearity(self,input):
        return input

    def computeVectorizedJacobian(self):
        return torch.ones(self.forwardOutput.shape)


class InvertibleOutputLayer(InvertibleLayer):
    """ Super class for the last layer of an invertible network, that will be
    trained using target propagation"""

    def __init__(self, inDim, layerDim, stepsize, lossFunction = 'mse'):
        super().__init__(inDim, layerDim, outDim=None, lossFunction =
        lossFunction)
        self.setStepsize(stepsize)


    def setStepsize(self, stepsize):
        if not isinstance(stepsize, float):
            raise TypeError("Expecting float number as stepsize, "
                            "got {}".format(type(stepsize)))
        if stepsize<=0.:
            raise ValueError("Expecting strictly positive stepsize")
        if stepsize > 0.5:
            raise RuntimeWarning("Stepsize bigger then 0.5 for setting output "
                                 "target can result in unexpected behaviour")
        self.stepsize = stepsize


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
            loss = lossFunction(self.forwardOutput.squeeze(), target_classes)
            return torch.mean(loss)#.numpy()
        elif self.lossFunction == 'mse':
            lossFunction = nn.MSELoss()
            loss = lossFunction(self.forwardOutput.squeeze(), target.squeeze())
            return torch.mean(loss)#.numpy()

    def propagateBackward(self, upperLayer):
        """ This function should never be called for an output layer,
        the backwardOutput should be set based on the
        loss of the layer with computeBackwardOutput"""
        raise NetworkError("Propagate Backward should never be called for an "
                           "output layer, use computeBackwardOutput "
                           "instead")

    def initBackwardParameters(self):
        """ Outputlayer does not have backward parameters"""
        pass

    def setBackwardParameters(self, backwardWeights, backwardBias):
        """ Outputlayer does not have backward parameters"""
        raise NetworkError("Outputlayer does not have backward parameters")

    def setBackwardGradients(self, backwardWeightsGrad, backwardBiasGrad):
        """ Outputlayer does not have backward parameters"""
        raise NetworkError("Outputlayer does not have backward parameters")

class InvertibleLinearOutputLayer(InvertibleOutputLayer):
    """ Invertible output layer with a linear activation function. This layer
    can so far only be combined with an mse loss
    function."""

    def forwardNonlinearity(self, linearActivation):
        return linearActivation

    def inverseNonlinearity(self,input):
        return input

    def computeVectorizedJacobian(self):
        return torch.ones(self.forwardOutput.shape)

    def computeBackwardOutput(self, target):
        """ Compute the backward output based on a small move from the
        forward output in the direction of the negative gradient of the loss
        function."""
        if not self.lossFunction == 'mse':
            raise NetworkError("a linear output layer can only be combined "
                               "with a mse loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension '
                             'x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        gradient = torch.mul(self.forwardOutput - target, 2)
        self.backwardOutput = self.forwardOutput - torch.mul(gradient,
                                                         self.stepsize)


class InvertibleInputLayer(InvertibleLayer):
    """ Input layer of the invertible neural network,
        e.g. the pixelvalues of a picture. """

    def __init__(self, layerDim, outDim, lossFunction = 'mse'):
        super().__init__(inDim=None, layerDim=layerDim, outDim=outDim,
                         lossFunction=lossFunction)

    def initForwardParameters(self):
        """ InputLayer has no forward parameters"""
        pass

    def initForwardParametersTilde(self):
        """ InputLayer has no forward parameters"""
        pass

    def initVelocities(self):
        """ InputLayer has no forward parameters"""
        raise RuntimeWarning("InputLayer has no forward parameters, so cannot "
                             "initialize velocities")
        pass

    def propagateForward(self, lowerLayer):
        """ This function should never be called for an input layer,
        the forwardOutput should be directly set
        to the input values of the network (e.g. the pixel values of a picture)
        """
        raise NetworkError("The forwardOutput should be directly set "
                           "to the input values of the network for "
                           "an InputLayer")


class InvertibleNetwork(Network):
    """ Invertible Network consisting of multiple invertible layers. This class
        provides a range of methods to facilitate training of the networks """

    def setLayers(self, layers):
        if not isinstance(layers, list):
            raise TypeError("Expecting a list object containing all the "
                            "layers of the network")
        if len(layers) < 2:
            raise ValueError("Expecting at least 2 layers (including input "
                             "and output layer) in a network")
        if not isinstance(layers[0], InvertibleInputLayer):
            raise TypeError("First layer of the network should be of type"
                            " InputLayer")
        if not isinstance(layers[-1], InvertibleOutputLayer):
            raise TypeError("Last layer of the network should be of "
                            "type OutputLayer")
        for i in range(1, len(layers)):
            if not isinstance(layers[i], InvertibleLayer):
                TypeError("All layers of the network should be of type "
                          "InvertibleLayer")
            if not layers[i - 1].layerDim == layers[i].inDim:
                raise ValueError("layerDim should match with inDim of "
                                 "next layer")
            if not layers[i-1].outDim == layers[i].layerDim:
                raise ValueError("outputDim should match with layerDim of next "
                                 "layer")

        self.layers = layers

    def propagateBackward(self, target):
        """ Propagate the layer targets backward
        through the network
        :param target: 3D tensor of size batchdimension x class dimension x 1
        """
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.layers[-1].forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: '
                             'batchdimension x class dimension x 1.'
                             ' Given target'
                             'has shape' + str(target.shape))

        self.layers[-1].computeBackwardOutput(target)
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].propagateBackward(self.layers[i + 1])
