from layers.invertible_layer import InvertibleInputLayer, \
    InvertibleLayer, InvertibleOutputLayer
from networks.bidirectional_network import BidirectionalNetwork


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
                            " InvertibleInputLayer")
        if not isinstance(layers[-1], InvertibleOutputLayer):
            raise TypeError("Last layer of the network should be of "
                            "type InvertibleOutputLayer")
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

    def save_inverse_error(self):
        for i in range(0, len(self.layers)-1):
            self.layers[i].save_inverse_error(self.layers[i+1])

    def save_state(self, global_step):
        super().save_state(global_step)
        self.save_inverse_error()