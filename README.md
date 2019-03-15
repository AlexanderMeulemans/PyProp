#PyProp
*A neural network toolbox for alternatives to backpropagation*

PyProp is a Python based toolbox for creating and training multilayer perceptron models with alternatives to backpropagation. It is based on the PyTorch framework and defines its own types of layers and networks (so it does not use the layers and networks of torch.nn) that are easily customizable to training methods other than backpropagation. Currently, both backpropagation and invertible target propagation are implemented, but it can easily be extended to other methods such as pure target propagation and difference target propagation, as long as there is a forward propagation and a backward propagation of training signals.
