"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from layers.layer import Layer, InputLayer, OutputLayer

class Network(object):
    """ Network consisting of multiple layers. This class provides a range of
    methods to facilitate training of the
    networks """

    def __init__(self, layers):
        """
        :param layers: list of all the layers in the network
        :param writer: SummaryWriter object to save states of the layer
        to tensorboard
        """
        super(Network, self).__init__()
        self.setLayers(layers)
        self.writer = self.layers[0].writer

    def setLayers(self, layers):
        if not isinstance(layers, list):
            raise TypeError("Expecting a list object containing all the "
                            "layers of the network")
        if len(layers) < 2:
            raise ValueError("Expecting at least 2 layers (including input "
                             "and output layer) in a network")
        if not isinstance(layers[0], InputLayer):
            raise TypeError("First layer of the network should be of type"
                            " InputLayer")
        if not isinstance(layers[-1], OutputLayer):
            raise TypeError("Last layer of the network should be of "
                            "type OutputLayer")
        for i in range(1, len(layers)):
            if not isinstance(layers[i], Layer):
                TypeError("All layers of the network should be of type Layer")
            if not layers[i - 1].layerDim == layers[i].inDim:
                raise ValueError("layerDim should match with inDim of "
                                 "next layer")

        self.layers = layers

    def initVelocities(self):
        """ Initialize the gradient velocities in all the layers. Only called
        when an optimizer with momentum is used."""
        for i in range(1, len(self.layers)):
            self.layers[i].initVelocities()

    def propagateForward(self, inputBatch):
        """ Propagate the inputbatch forward through the network
        :param inputBatch: Inputbatch of dimension
        batch dimension x input dimension x 1"""
        self.layers[0].setForwardOutput(inputBatch)
        for i in range(1, len(self.layers)):
            self.layers[i].propagateForward(self.layers[i - 1])

    def propagateBackward(self, target):
        """ Propagate the gradient of the loss function with respect to the
        linear activation of each layer backward
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
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].propagateBackward(self.layers[i + 1])

    def computeForwardGradients(self):
        """compute the gradient of the loss function to the
        parameters of each layer"""
        for i in range(1, len(self.layers)):
            self.layers[i].computeForwardGradients(self.layers[i - 1])

    def computeForwardGradientVelocities(self, momentum, learningRate):
        """Compute the gradient velocities for each layer"""
        for i in range(1, len(self.layers)):
            self.layers[i].computeForwardGradientVelocities(self.layers[i - 1],
                                                     momentum, learningRate)

    def computeGradients(self):
        self.computeForwardGradients()

    def computeGradientVelocities(self, momentum, learningRate):
        self.computeForwardGradientVelocities(momentum, learningRate)

    def updateForwardParametersWithVelocity(self):
        """ Update all the parameters of the network with the
                computed gradients velocities"""
        for i in range(1, len(self.layers)):
            self.layers[i].updateForwardParametersWithVelocity()

    def updateForwardParameters(self, learningRate):
        """ Update all the parameters of the network with the
        computed gradients"""
        for i in range(1, len(self.layers)):
            self.layers[i].updateForwardParameters(learningRate)

    def updateParameters(self, learningRate):
        self.updateForwardParameters(learningRate)

    def updateParametersWithVelocity(self):
        self.updateForwardParametersWithVelocity()

    def loss(self, target):
        """ Return the loss of each sample in the batch compared to
        the provided targets.
        :param target: 3D tensor of size batchdimension x class dimension x 1"""
        return self.layers[-1].loss(target)

    def zeroGrad(self):
        """ Set all the gradients of the network to zero"""
        for layer in self.layers:
            layer.zeroGrad()

    def predict(self, inputBatch):
        """ Return the networks predictions on a given input batch"""
        self.propagateForward(inputBatch)
        return self.layers[-1].forwardOutput

    def accuracy(self, targets):
        """ Return the test accuracy of network based on the given input
        test batch and the true targets
        IMPORTANT: first you have to run self.predict(inputBatch) in order
        to save the predictions in the output
        layer.
        IMPORTANT: the accuracy can only be computed for classification
        problems, thus the last layer should be
        a softmax """
        return self.layers[-1].accuracy(targets)

    def getOutput(self):
        return self.layers[-1].forwardOutput

    def setGlobalStep(self,global_step):
        for layer in self.layers:
            layer.global_step = global_step

    def save_state_histograms(self, global_step):
        self.setGlobalStep(global_step=global_step)
        for layer in self.layers:
            layer.save_state_histograms()

    def save_state(self, global_step):
        self.setGlobalStep(global_step)
        for layer in self.layers:
            layer.save_state()