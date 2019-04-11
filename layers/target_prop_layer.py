"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
import torch.nn as nn
from utils import helper_functions as hf
from utils.helper_classes import NetworkError, NotImplementedError
from layers.layer import Layer
from layers.bidirectional_layer import BidirectionalLayer

class TargetPropLayer(BidirectionalLayer):
    """ Target propagation with approximate inverses, but still the right
    form of the inverse."""
    pass

