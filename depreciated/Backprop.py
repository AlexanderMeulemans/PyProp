import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import helper_classes as hc, helper_functions as hf


class Layer():
    def __init__(self, inDim, layerDim, outDim, forwardNonLin='relu',
                 backwardNonLin='linear', learningRate=0.01):
        self.layerDim = layerDim
        self.inDim = inDim
        self.outDim = outDim
        self.forwardNonLin = forwardNonLin
        self.backwardNonLin = backwardNonLin
        self.initParameters()
        self.learningRate = learningRate

    def initParameters(self):
        self.forwardWeights = torch.rand(self.layerDim, self.inDim)
        self.forwardBias = torch.zeros(self.layerDim, 1)
        self.forwardWeightsGrad = torch.zeros(self.layerDim, self.inDim)
        self.forwardBiasGrad = torch.zeros(self.layerDim, 1)
        self.backwardWeights = torch.rand(self.layerDim, self.outDim)
        self.backwardBias = torch.zeros(self.layerDim, 1)
        self.backwardWeightsGrad = torch.zeros(self.layerDim, self.outDim)
        self.backwardBiasGrad = torch.zeros(self.layerDim, 1)

    def setForwardParameters(self, forwardWeights, forwardBias):
        assert hf.contains_no_nans(forwardWeights)
        assert hf.contains_no_nans(forwardBias)
        self.forwardWeights = forwardWeights
        self.forwardBias = forwardBias

    def setBackwardParametres(self, backwardWeights, backwardBias):
        assert hf.contains_no_nans(backwardWeights)
        assert hf.contains_no_nans(backwardBias)
        self.backwardWeights = backwardWeights
        self.backwardBias = backwardBias

    def zeroGrad(self):
        self.forwardWeightsGrad = torch.zeros(self.layerDim, self.inDim)
        self.forwardBiasGrad = torch.zeros(self.layerDim, 1)

    def propagateForward(self, lowerLayer):
        """
        :param lowerLayer: The first layer upstream of the layer 'self'
        :type lowerLayer: Layer
        :return saves the computed output of the layer to self.forward_output.
                forward_output is a 3D tensor of size batchDimension x layerDimension x 1
        """
        self.forwardInput = lowerLayer.forwardOutput
        self.linearActivation = torch.matmul(self.forwardWeights,
                                             self.forwardInput) + self.forwardBias
        if self.forwardNonLin == 'relu':
            self.forwardOutput = F.relu(self.linearActivation)
        elif self.forwardNonLin == 'linear':
            self.forwardOutput = self.linearActivation
        elif self.forwardNonLin == 'softmax':
            softmax = torch.nn.Softmax(1)
            self.forwardOutput = softmax(self.linearActivation)
        else:
            print("Error cause: ", str(self.forwardNonLin),
                  ' is not defined as an output nonlinearity')
            sys.exit(
                "Forward error: BPNet class is expecting different output nonlinearity")

    def propagateBackward(self, upperLayer):
        """
        :param upperLayer: the layer one step downstream of the layer 'self'
        :type upperLayer: Layer
        :return: saves the backwards output in self. backward_output is of size batchDimension x layerdimension  x 1
        """
        self.backwardInput = upperLayer.backwardOutput
        if self.forwardNonLin == 'relu':
            activationDer = torch.tensor(
                [[[1.] if self.linearActivation[i, j, 0] > 0 else [0.]
                  for j in range(self.linearActivation.size(1))] for i in
                 range(self.linearActivation.size(0))])
            self.backwardOutput = torch.mul(
                torch.matmul(torch.transpose(upperLayer.forwardWeights, -1, -2),
                             self.backwardInput), activationDer)
        elif self.forwardNonLin == 'linear':
            self.backwardOutput = torch.matmul(
                torch.transpose(upperLayer.forwardWeights, -1, -2),
                self.backwardInput)

        elif self.forwardNonLin == 'softmax':
            softmaxActivations = self.forwardOutput
            activationDer = torch.tensor([[[softmaxActivations[i, j, 0] * (
                    hf.kronecker(j, k) - softmaxActivations[i, k, 0])
                                            for k in
                                            range(softmaxActivations.size(1))]
                                           for j in
                                           range(softmaxActivations.size(1))]
                                          for i in
                                          range(softmaxActivations.size(0))])
            self.backwardOutput = torch.matmul(
                torch.transpose(activationDer, -1, -2),
                torch.matmul(torch.transpose(upperLayer.forwardWeights, -1, -2)
                             , self.backwardInput))
        else:
            print("Error cause: ", str(self.forwardNonLin),
                  ' is not defined as a forward nonlinearity')
            sys.exit(
                "Backward error: BPNet class is expecting different output nonlinearity")

        # construct 3D tensor with on first dimension the different batch samples, and on the second and third dimension
        # the adjusted weight matrix (element wise multiplication of entire column with the derivative evaluated in the
        # linear activation

        return

    def updateForwardParameters(self):
        forwardWeights = self.forwardWeights - torch.mul(
            self.forwardWeightsGrad, self.learningRate)
        forwardBias = self.forwardBias - torch.mul(self.forwardBiasGrad,
                                                   self.learningRate)
        self.setForwardParameters(forwardWeights, forwardBias)

    def computeGradients(self, lowerLayer):
        """
        :param lowerLayer: first layer upstream of the layer self
        :type lowerLayer: Layer
        :return: saves the gradients of the cost function to the layer parameters for all the batch samples

        """

        weight_gradients = torch.matmul(self.backwardOutput, torch.transpose(
            lowerLayer.forwardOutput, -1, -2))
        bias_gradients = self.backwardOutput
        self.forwardWeightsGrad = torch.mean(weight_gradients, 0)
        self.forwardBiasGrad = torch.mean(bias_gradients, 0)


class Network():
    def __init__(self, architecture, loss='crossEntropy', forwardNonLin='relu',
                 backwardNonLin='linear',
                 outputNonLin='softmax', learningRate=0.01):
        """
        :param architecture: list containing integers denoting the dimensions of the layers of the network.
                            The first entry is the input dimension of the network, the last entry the output dimension
        :param learningRate: learning rate for updating the parameters
        :type architecture: list
        """
        self.layers = []
        self.layers.append(
            Layer(architecture[0], architecture[0], architecture[1],
                  forwardNonLin=forwardNonLin,
                  backwardNonLin=backwardNonLin, learningRate=learningRate))
        for i in range(len(architecture) - 2):
            self.layers.append(
                Layer(architecture[i], architecture[i + 1], architecture[i + 2],
                      forwardNonLin=forwardNonLin,
                      backwardNonLin=backwardNonLin, learningRate=learningRate))

        self.layers.append(
            Layer(architecture[-2], architecture[-1], architecture[-1],
                  forwardNonLin=outputNonLin,
                  backwardNonLin='linear', learningRate=learningRate))
        self.lossFunction = loss
        self.loss = torch.tensor([])
        self.accuracyLst = torch.tensor([])

    def propagateForward(self, inputBatch):
        """
        
        :param inputBatch: 3D tensor with size batchdimension x inputdimension x 1
        :return: 
        """
        self.layers[0].forwardOutput = inputBatch
        for i in range(1, len(self.layers)):
            self.layers[i].propagateForward(self.layers[i - 1])

    def propagateBackward(self, target):
        """
        Compute loss and propagate gradients back through the network
        :param target:
        :return:
        """
        self.computeLoss(target)
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].propagateBackward(self.layers[i + 1])

    def computeLoss(self, target):
        """
        compute and save the loss and the derivative of the loss to the output of the network
        :param target: 3D tensor of size batchdimension x class dimension x 1
        :return:
        TODO: now it is assumed that cross entropy is always used in combination with a softmax layer and MSE in
        combination with a linear layer
        """
        lastLayer = self.layers[-1]
        target_classes = hf.prob2class(target)
        self.accuracyLst = torch.cat(
            (self.accuracyLst, torch.tensor([self.training_accuracy(target)])))
        if self.lossFunction == 'crossEntropy':
            lossFunction = nn.CrossEntropyLoss()
            loss = torch.empty(1)
            loss[0] = lossFunction(lastLayer.forwardOutput.squeeze(),
                                   target_classes)
            self.loss = torch.cat((self.loss, loss))
            self.layers[-1].backwardOutput = lastLayer.forwardOutput - target
        elif self.lossFunction == 'mse':
            lossFunction = nn.MSELoss()
            loss = torch.empty(1)
            loss[0] = lossFunction(lastLayer.forwardOutput.squeeze(), target)
            self.loss = torch.cat((self.loss, loss))
            self.layers[-1].backwardOutput = 2 * (
                    lastLayer.forwardOutput - target)
        else:
            print("Error cause: ", str(self.lossFunction),
                  ' is not defined as a loss function')
            sys.exit(
                "Loss error: Network class is expecting different output nonlinearity")

    def computeGradients(self):
        for i in range(1, len(self.layers)):
            self.layers[i].computeGradients(self.layers[i - 1])

    def updateParameters(self):
        for i in range(1, len(self.layers)):
            self.layers[i].updateForwardParameters()

    def batchTraining(self, batchInput, batchTarget):
        self.propagateForward(batchInput)
        self.propagateBackward(batchTarget)
        self.computeGradients()
        self.updateParameters()

    def zeroGrad(self):
        for layer in self.layers:
            layer.zeroGrad()

    def predict(self, inputBatch):
        self.propagateForward(inputBatch)
        return self.layers[-1].forwardOutput

    def test_loss(self, inputBatch, targets):
        predictions = self.predict(inputBatch)
        if self.lossFunction == 'crossEntropy':
            targets = hf.prob2class(targets)
            lossFunction = nn.CrossEntropyLoss()
            return lossFunction(predictions.squeeze(), targets)
        elif self.lossFunction == 'mse':
            lossFunction = nn.MSELoss()
            return lossFunction(predictions.squeeze(), targets.squeeze())
        else:
            print("Error cause: ", str(self.lossFunction),
                  ' is not defined as a loss function')
            sys.exit(
                "Loss error: Network class is expecting different output nonlinearity")

    def predict_class(self, inputBatch):
        self.propagateForward(inputBatch)
        class_probabilities = self.layers[-1].forwardOutput
        return hf.prob2class(class_probabilities)

    def accuracy(self, inputBatch, targets):
        predicted_classes = self.predict_class(inputBatch)
        return hf.accuracy(predicted_classes, hf.prob2class(targets))

    def training_accuracy(self, targets):
        class_probabilities = self.layers[-1].forwardOutput
        predicted_classes = hf.prob2class(class_probabilities)
        return hf.accuracy(predicted_classes, hf.prob2class(targets))

    def resetLoss(self):
        self.loss = torch.tensor([])
        self.accuracyLst = torch.tensor([])


class BPNet(nn.Module):
    def __init__(self, architecture, loss='crossEntropy', forwardNonLin='relu',
                 outputNonLin='softmax'):
        """

        :param architecture: list containing the dimension of each layer
        (first entry is the input dimension, last entry the output dimension)
        :param loss: Output loss function of the network
        :param forwardNonLin: layer nonlinearity for the forward pass
        :param outputNonLin: output nonlinearity
        :type architecture: List
        """
        super(BPNet, self).__init__()
        networkList = []
        for i in range(len(architecture) - 1):
            networkList.append(nn.Linear(architecture[i], architecture[i + 1]))
        if outputNonLin == 'softmax':
            networkList.append(nn.Softmax(2))
        elif outputNonLin == 'linear':
            networkList.append(hc.IdentityLayer())

        else:
            print("Error cause: ", str(outputNonLin),
                  ' is not defined as an output nonlinearity')
            sys.exit(
                "Init error: BPNet class is expecting different output nonlinearity")
        self.network = nn.ModuleList(networkList)

        self.forwardNonLin = forwardNonLin
        self.outputNonLin = outputNonLin
        self.loss = loss

    def forward(self, x):
        xLayers = []
        xLayers.append(x)
        if self.forwardNonLin == 'relu':
            for i in range(len(self.network) - 2):
                xLayers.append(F.relu(self.network[i](xLayers[-1])))
            xLayers.append(self.network[-2](xLayers[-1]))
            xLayers.append(self.network[-1](xLayers[-1]))
        else:
            print("Error cause: ", str(self.forwardNonLin),
                  " is not defined as a forward nonlinearity")
            sys.exit(
                "Forward error: BPNet class is expecting different forward nonlinearity")

        return xLayers

    def backward_custom(self, xLayers):
        return xLayers
