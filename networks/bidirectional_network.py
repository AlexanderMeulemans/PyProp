"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from networks.network import Network

class BidirectionalNetwork(Network):
    """ Bidirectional Network consisting of multiple layers that can propagate
        signals both forward and backward. This class
        provides a range of methods to facilitate training of the networks """


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

    def updateBackwardParameters(self, learningRate):
        """ Update all the parameters of the network with the
        computed gradients"""
        for i in range(0, len(self.layers)-1):
            self.layers[i].updateBackwardParameters(learningRate,
                                                    self.layers[i+1])

    def updateParameters(self, learningRate):
        self.updateForwardParameters(learningRate)
        self.updateBackwardParameters(learningRate)