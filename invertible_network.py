import torch
import torch.nn as nn
import torch.nn.functional as F
import HelperFunctions as hf
from HelperClasses import NetworkError, NotImplementedError
import numpy as np
from neuralnetwork import Layer, Network
from bidirectional_network import BidirectionalLayer, BidirectionalNetwork


class InvertibleLayer(BidirectionalLayer):
    """ Layer that is invertible to make it able to propagate exact targets."""

    def __init__(self,inDim, layerDim, outDim, lossFunction = 'mse',
                 name='invertible_layer'):
        if inDim is not None:
            if inDim < layerDim:
                raise ValueError("Expecting an input size bigger or equal to the "
                                 "layer size")
        if outDim is not None:
            if layerDim < outDim:
                raise ValueError("Expecting a layer size bigger or equal to the "
                                 "output size")
        super().__init__(inDim, layerDim, outDim, lossFunction, name=name)
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
        self.backwardBias = torch.empty(self.layerDim, 1)

    def initInverse(self, upperLayer):
        """ Initializes the backward weights to the inverse of the forward weights. After this
        initial inverse is computed, the sherman-morrison formula can be used to compute the
        inverses later in training"""
        self.backwardWeights = torch.inverse(torch.cat((upperLayer.forwardWeights,
                                            upperLayer.forwardWeightsTilde), 0))

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


    def setWeightUpdateU(self, u):
        """
        Save the u vector of the forward weight update to
         be able to use the Sherman-morrison formula
        ( https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula )
        """
        if not isinstance(u, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.u")
        self.u = u

    def setWeightUpdateV(self, v):
        """
        Save the v vector of the forward weight update to
         be able to use the Sherman-morrison formula
        ( https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula )
        """
        if not isinstance(v, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.u")
        self.v = v

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
            forwardOutputTilde = linearActivationTilde # no need to put through
            # nonlinearity, as in the backward pass, the inverse non-linearity
            # would be applied. Now we skip both to save computation
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

    def updateBackwardParameters(self, learningRate, upperLayer):
        """
        Update the backward weights and bias of the layer to resp.
        the inverse of the forward weights of the upperLayer and
        the negative bias of the upperLayer using the sherman-morrison
        formula
        """
        # take learning rate into u to apply Sherman-morrison formula later on
        u = torch.mul(upperLayer.u, learningRate)
        v = upperLayer.v
        if u.shape[0] < v.shape[0]:
            u = torch.cat((u, torch.zeros((v.shape[0]-u.shape[0],
                                           u.shape[1]))), 0)
        # apply Sherman-morrison formula to compute inverse

        denominator = 1 - torch.matmul(torch.transpose(v, -1, -2),
                                       torch.matmul(self.backwardWeights, u))
        numerator = torch.matmul(torch.matmul(self.backwardWeights, u),
                                 torch.matmul(torch.transpose(v, -1, -2),
                                              self.backwardWeights))
        backwardWeights = self.backwardWeights + torch.div(numerator,
                                                           denominator)
        backwardBias = - torch.cat((upperLayer.forwardBias,
                                    upperLayer.forwardBiasTilde), 0)
        self.setBackwardParameters(backwardWeights, backwardBias)



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

        target_inverse = upperLayer.inverseNonlinearity(
            upperLayer.backwardOutput)

        targetBar_inverse = torch.cat((target_inverse,
                               upperLayer.forwardOutputTilde), -2)

        backwardOutput = torch.matmul(self.backwardWeights,
                                      targetBar_inverse + self.backwardBias)
        self.setBackwardOutput(backwardOutput)

    def computeForwardGradients(self, lowerLayer):
        """
        :param lowerLayer: first layer upstream of the layer self
        :type lowerLayer: Layer
        :return: saves the gradients of the local cost function to the layer
        parameters for all the batch samples

        """
        if lowerLayer.forwardOutput.shape[0]>1:
            raise NetworkError('only batch sizes of size 1 are allowed,'
                               ' as otherwise the '
                               'inverse computation with sherman-morrisson'
                               ' is not possible')
        if self.lossFunction == 'mse':
            localLossDer = torch.mul(self.forwardOutput -
                                     self.backwardOutput, 2.)
        else:
            raise NetworkError("Expecting a mse local loss function")

        vectorizedJacobian = self.computeVectorizedJacobian()
        u = torch.mul(vectorizedJacobian, localLossDer)
        v = lowerLayer.forwardOutput

        weight_gradients = torch.matmul(u, torch.transpose(v, -1, -2))

        bias_gradients = u
        self.setWeightUpdateU(torch.reshape(u,(u.shape[-2],u.shape[-1])))
        self.setWeightUpdateV(torch.reshape(v,(v.shape[-2],v.shape[-1])))
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
        'mse', name='invertible_leaky_ReLU_layer'):
        super().__init__(inDim, layerDim, outDim, lossFunction, name=name)
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

    def __init__(self, inDim, layerDim, stepsize, lossFunction = 'mse',
                 name='invertible_output_layer'):
        super().__init__(inDim, layerDim, outDim=None, lossFunction =
        lossFunction, name=name)
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
            forwardOutputSqueezed = torch.reshape(self.forwardOutput,
                                                  (self.forwardOutput.shape[0],
                                                   self.forwardOutput.shape[1]))
            loss = lossFunction(forwardOutputSqueezed, target_classes)
            return torch.mean(loss)#.numpy()
        elif self.lossFunction == 'mse':
            lossFunction = nn.MSELoss()
            forwardOutputSqueezed = torch.reshape(self.forwardOutput,
                                                  (self.forwardOutput.shape[0],
                                                   self.forwardOutput.shape[1]))
            targetSqueezed = torch.reshape(target,
                                            (target.shape[0],
                                             target.shape[1]))
            loss = lossFunction(forwardOutputSqueezed, targetSqueezed)
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

class InvertibleSoftmaxOutputLayer(InvertibleOutputLayer):
    """ Invertible output layer with a linear activation function. This layer
    can so far only be combined with an mse loss
    function."""
    def __init__(self, inDim, layerDim, stepsize, lossFunction =
                 'crossEntropy', name='invertible_softmax_output_layer'):
        super().__init__(inDim, layerDim, stepsize=stepsize, lossFunction =
        lossFunction, name=name)
        self.normalization_constant = None

    def forwardNonlinearity(self, linearActivation):
        self.normalization_constant = torch.logsumexp(linearActivation,1)
        softmax = nn.Softmax(dim=1)
        print('original linear activation: {}'.format(linearActivation))
        return softmax(linearActivation)

    def inverseNonlinearity(self, input):
        print('computed linear activation: {}'.format(
            torch.log(input) + self.normalization_constant))
        return torch.log(input) + self.normalization_constant

    def computeVectorizedJacobian(self):
        raise NotImplementedError('Softmax outputlayer has a custom '
                                  'implementation of computeForwardGradients'
                                  'without the usage of '
                                  'computeVectorizedJacobian')

    def computeBackwardOutput(self, target):
        """ Compute the backward output based on a small move from the
        forward output in the direction of the negative gradient of the loss
        function."""
        if not self.lossFunction == 'crossEntropy':
            raise NetworkError("a softmax output layer can only be combined "
                               "with a crossEntropy loss")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension '
                             'x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        gradient = self.forwardOutput - target
        self.backwardOutput = self.forwardOutput - torch.mul(gradient,
                                                         self.stepsize)

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

    def computeForwardGradients(self, lowerLayer):
        """
        :param lowerLayer: first layer upstream of the layer self
        :type lowerLayer: Layer
        :return: saves the gradients of the local cost function to the layer
        parameters for all the batch samples

        """
        if lowerLayer.forwardOutput.shape[0]>1:
            raise NetworkError('only batch sizes of size 1 are allowed,'
                               ' as otherwise the '
                               'inverse computation with sherman-morrisson'
                               ' is not possible')
        if not self.lossFunction == 'crossEntropy':
            raise NetworkError('softmax ouptput layer should always be'
                               'combined with crossEntropy loss, now got {}'
                               'instead'.format(self.lossFunction))

        u = self.forwardOutput - self.backwardOutput
        v = lowerLayer.forwardOutput

        weight_gradients = torch.matmul(u, torch.transpose(v, -1, -2))

        bias_gradients = u
        self.setWeightUpdateU(torch.reshape(u,(u.shape[-2],u.shape[-1])))
        self.setWeightUpdateV(torch.reshape(v,(v.shape[-2],v.shape[-1])))
        self.setForwardGradients(torch.mean(weight_gradients, 0), torch.mean(
            bias_gradients, 0))

class InvertibleInputLayer(InvertibleLayer):
    """ Input layer of the invertible neural network,
        e.g. the pixelvalues of a picture. """

    def __init__(self, layerDim, outDim, lossFunction = 'mse',
                 name='invertible_input_layer'):
        super().__init__(inDim=None, layerDim=layerDim, outDim=outDim,
                         lossFunction=lossFunction,
                         name=name)

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


class InvertibleNetwork(BidirectionalNetwork):
    """ Invertible Network consisting of multiple invertible layers. This class
        provides a range of methods to facilitate training of the networks """
    def __init__(self, layers):
        super().__init__(layers)
        self.initInverses()

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

    def initInverses(self):
        """ Initialize the backward weights of all layers to the inverse of
        the forward weights of
        the layer on top."""
        for i in range(0, len(self.layers)-1):
            self.layers[i].initInverse(self.layers[i+1])


