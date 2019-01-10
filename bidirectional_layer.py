import torch
import torch.nn as nn
import torch.nn.functional as F
import HelperFunctions as hf
from HelperClasses import NetworkError
import numpy as np
from neuralnetwork import Layer, Network

class BidirectionalLayer(Layer):
    """ Layer in a neural network with feedforward weights as wel as
    feedbackward weights."""
    def __init__(self, inDim, layerDim, outDim, lossFunction = 'mse'):
        super().__init__(inDim, layerDim)
        if outDim is not None: # if the layer is an outputlayer, outDim is None
            self.setOutDim(outDim)
        self.initBackwardParameters()
        self.setLossFunction(lossFunction)


    def setOutDim(self, outDim):
        if not isinstance(outDim, int):
            raise TypeError("Expecting an integer layer dimension")
        if outDim <= 0:
            raise ValueError("Expecting strictly positive layer dimension")
        self.outDim = outDim

    def setLossFunction(self, lossFunction):
        if not isinstance(lossFunction, str):
            raise TypeError("Expecting a string to indicate loss function, "
                            "got {}".format(type(lossFunction)))
        if not (lossFunction == 'mse'):
            raise ValueError("Only the mse local loss function is defined "
                             "yet, got {}".format(lossFunction))
        self.lossFunction= lossFunction

    def initBackwardParameters(self):
        """ Initializes the layer parameters when the layer is created.
        This method should only be used when creating
        a new layer. Use setbackwardParameters to update the parameters and
        computeGradient to update the gradients"""
        self.backwardWeights = torch.rand(self.layerDim, self.outDim)
        self.backwardBias = torch.zeros(self.layerDim, 1)
        self.backwardWeightsGrad = torch.zeros(self.layerDim, self.outDim)
        self.backwardBiasGrad = torch.zeros(self.layerDim, 1)

    def setBackwardParameters(self, backwardWeights, backwardBias):
        if not isinstance(backwardWeights, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.backwardWeights")
        if not isinstance(backwardBias, torch.Tensor):
            raise TypeError("Expecting a tensor object for self.backwardBias")
        if hf.containsNaNs(backwardWeights):
            raise ValueError("backwardWeights contains NaNs")
        if hf.containsNaNs(backwardBias):
            raise ValueError("backwardBias contains NaNs")
        if not backwardWeights.shape == self.backwardWeights.shape:
            raise ValueError("backwardWeights has not the correct shape")
        if not backwardBias.shape == self.backwardBias.shape:
            raise ValueError("backwardBias has not the correct shape")

        self.backwardWeights = backwardWeights
        self.backwardBias = backwardBias

    def setBackwardGradients(self, backwardWeightsGrad, backwardBiasGrad):
        if not isinstance(backwardWeightsGrad, torch.Tensor):
            raise TypeError("Expecting a tensor object "
                            "for self.backwardWeightsGrad")
        if not isinstance(backwardBiasGrad, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.backwardBiasGrad")
        if hf.containsNaNs(backwardWeightsGrad):
            raise ValueError("backwardWeightsGrad contains NaNs")
        if hf.containsNaNs(backwardBiasGrad):
            raise ValueError("backwardBias contains NaNs")
        if not backwardWeightsGrad.shape == self.backwardWeightsGrad.shape:
            raise ValueError("backwardWeightsGrad has not the correct shape")
        if not backwardBiasGrad.shape == self.backwardBiasGrad.shape:
            raise ValueError("backwardBiasGrad has not the correct shape")

        self.backwardWeightsGrad = backwardWeightsGrad
        self.backwardBiasGrad = backwardBiasGrad

    def setBackwardOutput(self, backwardOutput):
        if not isinstance(backwardOutput, torch.Tensor):
            raise TypeError("Expecting a tensor object for "
                            "self.backwardOutput")
        if not backwardOutput.size(-2) == self.layerDim:
            raise ValueError("Expecting same dimension as layerDim")
        if not backwardOutput.size(-1) == 1:
            raise ValueError("Expecting same dimension as layerDim")
        self.backwardOutput = backwardOutput

    def backwardNonlinearity(self, linearActivation):
        """ This method should be always overwritten by the children"""
        raise NetworkError("The method backwardNonlinearity should always be "
                           "overwritten by children of Layer. Layer on itself "
                           "cannot be used in a network")










