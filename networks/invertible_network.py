"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

from layers.invertible_layer import InvertibleInputLayer, \
    InvertibleLayer, InvertibleOutputLayer
from networks.bidirectional_network import BidirectionalNetwork


class InvertibleNetwork(BidirectionalNetwork):
    """ Invertible Network consisting of multiple invertible layers. This class
        provides a range of methods to facilitate training of the networks """
    def __init__(self, layers, name=None, debug_mode=False):
        super().__init__(layers,name)
        self.initInverses()
        self.debug_mode = debug_mode

    def setLayers(self, layers):
        if not isinstance(layers, list):
            raise TypeError("Expecting a list object containing all the "
                            "layers of the network")
        if len(layers) < 2:
            raise ValueError("Expecting at least 2 layers (including input "
                             "and output layer) in a network")
        if not isinstance(layers[0], InvertibleInputLayer):
            raise TypeError("First layer of the network should be of type"
                            " InvertibleInputLayer")
        if not isinstance(layers[-1], InvertibleOutputLayer):
            raise TypeError("Last layer of the network should be of "
                            "type InvertibleOutputLayer")
        for i in range(1, len(layers)):
            if not isinstance(layers[i], InvertibleLayer):
                TypeError("All layers of the network should be of type "
                          "InvertibleLayer")
            if not layers[i - 1].layerDim == layers[i].inDim:
                raise ValueError("layer_dim should match with in_dim of "
                                 "next layer")
            if not layers[i-1].outDim == layers[i].layerDim:
                raise ValueError("outputDim should match with layer_dim of next "
                                 "layer")

        self.layers = layers

    def initInverses(self):
        """ Initialize the backward weights of all layers to the inverse of
        the forward weights of
        the layer on top."""
        for i in range(0, len(self.layers)-1):
            self.layers[i].initInverse(self.layers[i+1])

    def save_inverse_error(self):
        for i in range(0, len(self.layers)-1):
            self.layers[i].save_inverse_error(self.layers[i+1])

    def save_state(self, global_step):
        super().save_state(global_step)
        self.save_inverse_error()

    def test_invertibility(self, inputBatch):
        """ Propagate an input batch forward and backward, and compute the error
        of the inversions (backpropagated targets should be equal to forward
        activations"""
        self.propagateForward(inputBatch)
        self.custom_propagate_backward(self.layers[-1].forwardOutput)
        for layer in self.layers:
            layer.save_invertibility_test()

    def custom_propagate_backward(self, backward_input):
        """
        Propagate directly the given backward_input backwards through the
        network instead of computing the output target value with
        computeBackwardOutput().
        """
        self.layers[-1].setBackwardOutput(backward_input)
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].propagateBackward(self.layers[i + 1])

    def save_state(self, global_step):
        """ Also perform an invertibiltiy test at the end of each batch
        if debug mode is on True"""
        super().save_state(global_step)
        self.test_invertibility(self.layers[0].forwardOutput)
