import torch
import torch.nn as nn
import torch.nn.functional as F
import HelperFunctions as hf
from HelperClasses import NetworkError
import numpy as np

class Layer(object):
    """ Parent class of all occuring layers in neural networks with only feedforward weights.
    This class should not be used directly, only via its children"""

    def __init__(self, inDim, layerDim):
        """
        Initializes the Layer object
        :param inDim: input dimension of the layer (equal to the layer dimension of the previous layer in the network)
        :param layerDim: Layer dimension
        """
        self.setLayerDim(layerDim)
        self.setInDim(inDim)
        self.initParameters()

    def setLayerDim(self,layerDim):
        if not isinstance(layerDim,int):
            raise TypeError("Expecting an integer layer dimension")
        if layerDim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.layerDim = layerDim

    def setInDim(self,inDim):
        if not isinstance(inDim,int):
            raise TypeError("Expecting an integer layer dimension")
        if inDim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.inDim = inDim

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

        self.forwardWeights = forwardWeights
        self.forwardBias = forwardBias

    def setForwardGradients(self, forwardWeightsGrad, forwardBiasGrad):
        if not isinstance(forwardWeightsGrad, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forwardWeightsGrad")
        if not isinstance(forwardBiasGrad, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forwardBiasGrad")
        if hf.containsNaNs(forwardWeightsGrad):
            raise ValueError("forwardWeightsGrad contains NaNs")
        if hf.containsNaNs(forwardBiasGrad):
            raise ValueError("forwardBias contains NaNs")
        if not forwardWeightsGrad.shape == self.forwardWeightsGrad.shape:
            raise ValueError("forwardWeightsGrad has not the correct shape")
        if not forwardBiasGrad.shape == self.forwardBiasGrad.shape:
            raise ValueError("forwardBiasGrad has not the correct shape")

        self.forwardWeightsGrad = forwardWeightsGrad
        self.forwardBiasGrad = forwardBiasGrad

    def setForwardOutput(self,forwardOutput):
        if not isinstance(forwardOutput, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.forwardOutput")
        if not forwardOutput.size(-2) == self.layerDim:
            raise ValueError("Expecting same dimension as layerDim")
        if not forwardOutput.size(-1) == 1:
            raise ValueError("Expecting same dimension as layerDim")
        self.forwardOutput = forwardOutput

    def setBackwardOutput(self,backwardOutput):
        if not isinstance(backwardOutput, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.backwardOutput")
        if not backwardOutput.size(-2) == self.layerDim:
            raise ValueError("Expecting same dimension as layerDim")
        if not backwardOutput.size(-1) == 1:
            raise ValueError("Expecting same dimension as layerDim")
        self.backwardOutput = backwardOutput

    def initParameters(self):
        """ Initializes the layer parameters when the layer is created. This method should only be used when creating
        a new layer. Use setForwardParameters to update the parameters and computeGradient to update the gradients"""
        self.forwardWeights = torch.rand(self.layerDim, self.inDim)
        self.forwardBias = torch.zeros(self.layerDim, 1)
        self.forwardWeightsGrad = torch.zeros(self.layerDim, self.inDim)
        self.forwardBiasGrad = torch.zeros(self.layerDim, 1)


    def zeroGrad(self):
        """ Set the gradients of the layer parameters to zero """
        self.forwardWeightsGrad = torch.zeros(self.layerDim, self.inDim)
        self.forwardBiasGrad = torch.zeros(self.layerDim,1)

    def updateForwardParameters(self, learningRate):
        """
        Update the forward weights and bias of the layer using the computed gradients
        :param learningRate: Learning rate of the layer
        """
        if not isinstance(learningRate,float):
            raise TypeError("Expecting a float number as learningRate")
        if learningRate <= 0.:
            raise ValueError("Expecting a strictly positive learningRate")

        forwardWeights = self.forwardWeights - torch.mul(self.forwardWeightsGrad, learningRate)
        forwardBias = self.forwardBias - torch.mul(self.forwardBiasGrad, learningRate)
        self.setForwardParameters(forwardWeights, forwardBias)

    def propagateForward(self,lowerLayer):
        """
        :param lowerLayer: The first layer upstream of the layer 'self'
        :type lowerLayer: Layer
        :return saves the computed output of the layer to self.forwardOutput.
                forwardOutput is a 3D tensor of size batchDimension x layerDimension x 1
        """
        if not isinstance(lowerLayer,Layer):
            raise TypeError("Expecting a Layer object as argument for propagateForward")
        if not lowerLayer.layerDim == self.inDim:
            raise ValueError("Layer sizes are not compatible for propagating forward")

        self.forwardInput = lowerLayer.forwardOutput
        self.linearActivation = torch.matmul(self.forwardWeights,self.forwardInput) + self.forwardBias
        self.forwardOutput = self.nonlinearity(self.linearActivation)

    def nonlinearity(self,linearActivation):
        """ This method should be always overwritten by the children"""
        raise NetworkError("The method nonlinearity should always be overwritten by children of Layer. Layer on itself "
                        "cannot be used in a network")

    def computeGradients(self, lowerLayer):
        """
        :param lowerLayer: first layer upstream of the layer self
        :type lowerLayer: Layer
        :return: saves the gradients of the cost function to the layer parameters for all the batch samples

        """

        weight_gradients = torch.matmul(self.backwardOutput,torch.transpose(lowerLayer.forwardOutput,-1,-2))
        bias_gradients = self.backwardOutput
        self.setForwardGradients(torch.mean(weight_gradients, 0),torch.mean(bias_gradients, 0))


class ReluLayer(Layer):
    """ Layer of a neural network with a RELU activation function"""

    def nonlinearity(self,linearActivation):
        """ Returns the nonlinear activation of the layer"""
        return F.relu(linearActivation)

    def propagateBackward(self,upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backwardOutput is of size batchDimension x layerdimension  x 1
        """
        if not isinstance(upperLayer,Layer):
            raise TypeError("Expecting a Layer object as argument for propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible for propagating backwards")

        self.backwardInput = upperLayer.backwardOutput
        # Construct vectorized Jacobian for all batch samples.
        activationDer = torch.tensor([[[1.] if self.linearActivation[i,j,0]>0 else [0.]
                            for j in range(self.linearActivation.size(1))]
                            for i in range(self.linearActivation.size(0))])
        backwardOutput = torch.mul(torch.matmul(torch.transpose(upperLayer.forwardWeights,-1,-2),
                                                     self.backwardInput),activationDer)
        self.setBackwardOutput(backwardOutput)


class SoftmaxLayer(Layer):
    """ Layer of a neural network with a Softmax activation function"""

    def nonlinearity(self,linearActivation):
        """ Returns the nonlinear activation of the layer"""
        softmax = torch.nn.Softmax(1)
        return softmax(linearActivation)

    def propagateBackward(self,upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backwardOutput is of size batchDimension x layerdimension  x 1
        """
        if not isinstance(upperLayer,Layer):
            raise TypeError("Expecting a Layer object as argument for propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible for propagating backwards")

        self.backwardInput = upperLayer.backwardOutput
        # Construct Jacobian for all batch samples.
        softmaxActivations = self.forwardOutput
        jacobian = torch.tensor([[[softmaxActivations[i, j, 0] * (hf.kronecker(j, k) - softmaxActivations[i, k, 0])
                                        for k in range(softmaxActivations.size(1))]
                                       for j in range(softmaxActivations.size(1))]
                                      for i in range(softmaxActivations.size(0))])
        backwardOutput = torch.matmul(torch.transpose(jacobian, -1, -2),
                                           torch.matmul(torch.transpose(upperLayer.forwardWeights, -1, -2)
                                                        , self.backwardInput))
        self.setBackwardOutput(backwardOutput)

class LinearLayer(Layer):
    """ Layer of a neural network with a linear activation function"""

    def nonlinearity(self,linearActivation):
        """ Returns the nonlinear activation of the layer"""
        return linearActivation

    def propagateBackward(self,upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backwardOutput is of size batchDimension x layerdimension  x 1
        """
        if not isinstance(upperLayer,Layer):
            raise TypeError("Expecting a Layer object as argument for propagateBackward")
        if not upperLayer.inDim == self.layerDim:
            raise ValueError("Layer sizes are not compatible for propagating backwards")

        self.backwardInput = upperLayer.backwardOutput
        backwardOutput = torch.matmul(torch.transpose(upperLayer.forwardWeights, -1, -2), self.backwardInput)
        self.setBackwardOutput(backwardOutput)

class InputLayer(Layer):
    """ Input layer of the neural network, e.g. the pixelvalues of a picture. """

    def __init__(self,layerDim):
        """ InputLayer has only a layerDim and a forward activation that can be set,
         no input dimension nor parameters"""
        self.setLayerDim(layerDim)


    def propagateForward(self, lowerLayer):
        """ This function should never be called for an input layer, the forwardOutput should be directly set
        to the input values of the network (e.g. the pixel values of a picture) """
        raise NetworkError("The forwardOutput should be directly set to the input values of the network for "
                           "an InputLayer")

    def propagateBackward(self, upperLayer):
        """ This function should never be called for an input layer, there is no point in having a backward output
         here, as this layer has no parameters to update"""
        raise NetworkError("Propagate Backward should never be called for an input layer, there is no point in having "
                           "a backward output here, as this layer has no parameters to update")

class OutputLayer(Layer):
    """" Super class for the last layer of a network. This layer has a loss as extra attribute and some extra
    methods as explained below. """

    def __init__(self,inDim, layerDim, lossFunction):
        """
        :param inDim: input dimension of the layer, equal to the layer dimension of the second last layer in the network
        :param layerDim: Layer dimension
        :param loss: string indicating which loss function is used to compute the network loss
        """
        super().__init__(inDim, layerDim)
        self.setLossFunction(lossFunction)

    def setLossFunction(self,lossFunction):
        if not isinstance(lossFunction, str):
            raise TypeError('Expecting a string as indicator for the loss function')
        if not (lossFunction == 'mse' or lossFunction == 'crossEntropy'):
            raise NetworkError('Expecting an mse or crossEntropy loss')
        self.lossFunction = lossFunction

    def loss(self, target):
        """ Compute the loss with respect to the target
        :param target: 3D tensor of size batchdimension x class dimension x 1
        """
        if not isinstance(target,torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        if self.lossFunction == 'crossEntropy':
            # Convert output 'class probabilities' to one class per batch sample (with highest class probability)
            target_classes = hf.prob2class(target)
            lossFunction = nn.CrossEntropyLoss()
            loss = lossFunction(self.forwardOutput.squeeze(), target_classes)
            return torch.mean(loss).numpy()
        elif self.lossFunction == 'mse':
            lossFunction = nn.MSELoss()
            loss = lossFunction(self.forwardOutput.squeeze(), target.squeeze())
            return torch.mean(loss).numpy()

    def propagateBackward(self, upperLayer):
        """ This function should never be called for an output layer, the backwardOutput should be set based on the
        loss of the layer with computeBackwardOutput"""
        raise NetworkError("Propagate Backward should never be called for an output layer, use computeBackwardOutput "
                           "instead")


class SoftmaxOutputLayer(OutputLayer):
    """ Output layer with a softmax activation function. This layer should always be combined with a crossEntropy
    loss."""

    def nonlinearity(self,linearActivation):
        """ Returns the nonlinear activation of the layer"""
        softmax = torch.nn.Softmax(1)
        return softmax(linearActivation)

    def computeBackwardOutput(self,target):
        """ Compute the backward output based on the derivative of the loss to the linear activation of this layer
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        if not self.lossFunction == 'crossEntropy':
            raise NetworkError("a softmax output layer should always be combined with a cross entropy loss")
        if not isinstance(target,torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        
        backwardOutput = self.forwardOutput - target
        self.setBackwardOutput(backwardOutput)

    def propagateForward(self,lowerLayer):
        """ Normal forward propagation, but on top of that, save the predicted classes in self."""
        super().propagateForward(lowerLayer)
        self.predictedClasses = hf.prob2class(self.forwardOutput)

    def accuracy(self, target):
        """ Compute the accuracy if the network predictions with respect to the given true targets.
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        if not isinstance(target,torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        return hf.accuracy(self.predictedClasses, hf.prob2class(target))


        
class LinearOutputLayer(OutputLayer):
    """ Output layer with a linear activation function. This layer can so far only be combined with an mse loss
    function."""
    
    def nonlinearity(self,linearActivation):
        """ Returns the nonlinear activation of the layer"""
        return linearActivation
    
    def computeBackwardOutput(self,target):
        """ Compute the backward output based on the derivative of the loss to the linear activation of this layer"""
        if not self.lossFunction == 'mse':
            raise NetworkError("a linear output layer can only be combined with a mse loss")
        if not isinstance(target,torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension x class dimension x 1. Given target'
                             'has shape' + str(target.shape))
        backwardOutput = 2*(self.forwardOutput - target)
        self.setBackwardOutput(backwardOutput)
        

class Network(object):
    """ Network consisting of multiple layers. This class provides a range of methods to facilitate training of the
    networks """
    
    def __init__(self, layers):
        """
        :param layers: list of all the layers in the network
        """
        self.setLayers(layers)
        
    def setLayers(self,layers):
        if not isinstance(layers,list):
            raise TypeError("Expecting a list object containing all the layers of the network")
        if len(layers) < 2:
            raise ValueError("Expecting at least 2 layers (including input and output layer) in a network")
        if not isinstance(layers[0],InputLayer):
            raise TypeError("First layer of the network should be of type InputLayer")
        if not isinstance(layers[-1],OutputLayer):
            raise TypeError("Last layer of the network should be of type OutputLayer")
        for i in range(1,len(layers)):
            if not isinstance(layers[i],Layer):
                TypeError("All layers of the network should be of type Layer")
            if not layers[i-1].layerDim == layers[i].inDim:
                raise ValueError("layerDim should match with inDim of next layer")

        self.layers = layers

    def propagateForward(self, inputBatch):
        """ Propagate the inputbatch forward through the network
        :param inputBatch: Inputbatch of dimension batch dimension x input dimension x 1"""
        self.layers[0].setForwardOutput(inputBatch)
        for i in range(1,len(self.layers)):
            self.layers[i].propagateForward(self.layers[i-1])

    def propagateBackward(self, target):
        """ Propagate the gradient of the loss function with respect to the linear activation of each layer backward
        through the network and compute the gradient of the loss function to the parameters of each layers
        :param target: 3D tensor of size batchdimension x class dimension x 1 """
        if not isinstance(target,torch.Tensor):
            raise TypeError("Expecting a torch.Tensor object as target")
        if not self.layers[-1].forwardOutput.shape == target.shape:
            raise ValueError('Expecting a tensor of dimensions: batchdimension x class dimension x 1. Given target'
                             'has shape' + str(target.shape))

        self.layers[-1].computeBackwardOutput(target)
        self.layers[-1].computeGradients(self.layers[-2])
        for i in range(len(self.layers)-2,0,-1):
            self.layers[i].propagateBackward(self.layers[i+1])
            self.layers[i].computeGradients(self.layers[i-1])

    def updateParameters(self, learningRate):
        """ Update all the parameters of the network with the computed gradients"""
        for i in range(1,len(self.layers)):
            self.layers[i].updateForwardParameters(learningRate)

    def loss(self,target):
        """ Return the loss of each sample in the batch compared to the provided targets.
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        return self.layers[-1].loss(target)

    def batchTraining(self,batchInput,target,learningRate):
        """ Perfrom a complete batch training with the given input batch and targets"""

        self.propagateForward(batchInput)
        self.propagateBackward(target)
        self.updateParameters(learningRate)

    def zeroGrad(self):
        """ Set all the gradients of the network to zero"""
        for layer in self.layers:
            layer.zeroGrad()

    def predict(self, inputBatch):
        """ Return the networks predictions on a given input batch"""
        self.propagateForward(inputBatch)
        return self.layers[-1].forwardOutput

    def accuracy(self,targets):
        """ Return the test accuracy of network based on the given input test batch and the true targets
        IMPORTANT: first you have to run self.predict(inputBatch) in order to save the predictions in the output
        layer.
        IMPORTANT: the accuracy can only be computed for classification problems, thus the last layer should be
        a softmax """
        return self.layers[-1].accuracy(targets)


class Optimizer(object):
    """" Super class for all the different optimizers (e.g. SGD)"""

    def __init__(self, network, computeAccuracies = False):
        """
        :param network: network to train
        :param computeAccuracies: True if the optimizer should also save the accuracies. Only possible with
        classification problems
        :type network: Network
        """
        self.epochLosses = np.array([])
        self.batchLosses = np.array([])
        self.singleBatchLosses = np.array([])
        self.setNetwork(network)
        self.setComputeAccuracies(computeAccuracies)
        if self.computeAccuracies:
            self.epochAccuracies = np.array([])
            self.batchAccuracies = np.array([])
            self.singleBatchAccuracies = np.array([])

    def setNetwork(self,network):
        if not isinstance(network, Network):
            raise TypeError("Expecting Network object, instead got {}".format(type(network)))
        self.network = network

    def setComputeAccuracies(self,computeAccuracies):
        if not isinstance(computeAccuracies,bool):
            raise TypeError("Expecting a bool as computeAccuracies")
        self.computeAccuracies = computeAccuracies

    def resetSingleBatchLosses(self):
        self.singleBatchLosses = np.array([])

    def resetSingleBatchAccuracies(self):
        self.singleBatchAccuracies = np.array([])

    def runMNIST(self,trainLoader):
        """ Train the network on the total training set of MNIST as long as epoch loss is above the threshold
        :param trainLoader: a torch.utils.data.DataLoader object which containts the dataset"""
        if not isinstance(trainLoader, torch.utils.data.DataLoader):
            raise TypeError("Expecting a DataLoader object, now got a {}".format(type(trainLoader)))
        epoch = 0
        epochLoss = float('inf')
        batchSize = trainLoader.batch_size
        while epochLoss > self.threshold:
            for batch_idx, (data, target) in enumerate(trainLoader):
                if batch_idx % 10 == 0:
                    print('batch: ' + str(batch_idx))
                data = data.view(-1, 28*28, 1)
                target = hf.oneHot(target, 10)
                self.step(data, target)
            epochLoss = np.mean(self.singleBatchLosses)
            self.resetSingleBatchLosses()
            self.epochLosses = np.append(self.epochLosses, epochLoss)
            epoch += 1
            print('Epoch: ' + str(epoch) + ' ------------------------')
            print('Loss: ' + str(epochLoss))
            if self.computeAccuracies:
                epochAccuracy = np.mean(self.singleBatchAccuracies)
                self.epochAccuracies = np.append(self.epochAccuracies,epochAccuracy)
                self.resetSingleBatchAccuracies()
                print('Training Accuracy: ' + str(epochAccuracy))

        print('====== Training finished =======')


class SGD(Optimizer):
    """ Stochastic Gradient Descend optimizer"""

    def __init__(self, network, threshold, learningRate, computeAccuracies = False):
        """
        :param threshold: the optimizer will run until the network loss is below this threshold
        :param learningRate: Learningrate used to update the parameters with their gradients
        """
        super().__init__(network, computeAccuracies)
        self.setThreshold(threshold)
        self.setLearningRate(learningRate)

    def setLearningRate(self, learningRate):
        if not isinstance(learningRate,float):
            raise TypeError("Expecting a float number as learningRate")
        if learningRate <=0:
            raise ValueError("Expecting a strictly positive learning rate")
        self.learningRate = learningRate

    def setThreshold(self, threshold):
        if not isinstance(threshold, float):
            raise TypeError("Expecting a float number as threshold")
        if threshold <=0:
            raise ValueError("Expecting a strictly positive threshold")
        self.threshold = threshold

    def step(self, inputBatch, targets):
        """ Perform one batch optimizing step"""
        self.network.batchTraining(inputBatch,targets,self.learningRate)
        self.batchLosses = np.append(self.batchLosses, self.network.loss(targets))
        self.singleBatchLosses = np.append(self.singleBatchLosses,self.network.loss(targets))
        if self.computeAccuracies:
            self.batchAccuracies = np.append(self.batchAccuracies, self.network.accuracy(targets))
            self.singleBatchAccuracies = np.append(self.singleBatchAccuracies, self.network.accuracy(targets))
        

        










