"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from networks.network import Network


class GenerateDatasetFromModel(object):
    """ Generates a toy example dataset from a given true network, that can be
    used to train a new network"""

    def __init__(self, true_network):
        if not isinstance(true_network, Network):
            raise TypeError("Expecting Network object for true_network")
        self.true_network = true_network
        self.input_size = true_network.layers[0].layer_dim
        self.output_size = true_network.layers[-1].layer_dim

    def generate(self, nb_batches, batch_sizes):
        """ Generate dataset of given batch size and number of batches"""
        input_dataset = torch.randn(nb_batches, batch_sizes, self.input_size, 1)
        output_dataset = torch.empty(nb_batches, batch_sizes,
                                     self.output_size, 1)
        for i in range(nb_batches):
            self.true_network.propagate_forward(input_dataset[i, :, :, :])
            output_dataset[i, :, :, :] = self.true_network.get_output()
        return input_dataset, output_dataset
