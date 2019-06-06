# PyProp
*A neural network toolbox for alternatives to backpropagation*

PyProp is a Python based toolbox for creating and training multilayer perceptron models with alternatives to backpropagation. It is based on the PyTorch framework and defines its own types of layers and networks (so it does not use the layers and networks of torch.nn) that are easily customizable to training methods other than backpropagation. Currently, both backpropagation and invertible target propagation are implemented, but it can easily be extended to other methods such as pure target propagation and difference target propagation, as long as there is a forward propagation and a backward propagation of training signals.

## Reproducing figures of chapter 5
For reproducing the figures of chapter 5 of the master thesis of Alexander Meulemans, run the corresponding python files in the module 'figure_scripts'. Sometimes a script has to run multiple times with different parameter settings. In that case, documentation is added at the top of the file on how the parameter values need to be changed for reproducing each (sub-)figure.

## Contact
For futher questions on this toolbox, please contact Alexander: alexander.meulemans@gmail.com