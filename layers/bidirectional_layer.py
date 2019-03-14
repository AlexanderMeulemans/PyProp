import torch
from utils import HelperFunctions as hf
from utils.HelperClasses import NetworkError
from layers.layer import Layer

class BidirectionalLayer(Layer):
    """ Layer in a neural network with feedforward weights as well as
    feedbackward weights."""
    def __init__(self, inDim, layerDim, outDim, writer, lossFunction = 'mse',
                 name='bidirectional_layer'):
        super().__init__(inDim, layerDim, name=name, writer=writer)
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
        if not ((lossFunction == 'mse') or (lossFunction == 'crossEntropy')):
            raise ValueError("Only the mse or cross entropy"
                             " local loss function is defined "
                             "yet, got {}".format(lossFunction))
        self.lossFunction= lossFunction

    def initBackwardParameters(self):
        """ Initializes the layer parameters when the layer is created.
        This method should only be used when creating
        a new layer. Use setbackwardParameters to update the parameters and
        computeGradient to update the gradients"""
        self.backwardWeights = torch.randn(self.layerDim, self.outDim)
        self.backwardBias = torch.zeros(self.layerDim, 1)
        self.backwardWeightsGrad = torch.zeros(self.layerDim, self.outDim)
        self.backwardBiasGrad = torch.zeros(self.layerDim, 1)
        self.save_initial_backward_state()

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

    def updateBackwardParameters(self, learningRate, upperLayer):
        """ Should be implemented by the child classes"""
        raise NetworkError('This method should be overwritten by the '
                           'child classes')

    def save_backward_weights(self):
        weight_norm = torch.norm(self.backwardWeights)
        bias_norm = torch.norm(self.backwardBias)
        self.writer.add_scalar(tag='{}/backward_weights'
                                   '_norm'.format(self.name),
                               scalar_value=weight_norm,
                               global_step=self.global_step)
        self.writer.add_scalar(tag='{}/backward_bias'
                                   '_norm'.format(self.name),
                               scalar_value=bias_norm,
                               global_step=self.global_step)

    def save_backward_weights_hist(self):
        self.writer.add_histogram(tag='{}/backward_weights_'
                                      'hist'.format(
            self.name),
            values=self.backwardWeights,
            global_step=self.global_step)
        self.writer.add_histogram(tag='{}/backward_bias_'
                                      'hist'.format(
            self.name),
            values=self.backwardBias,
            global_step=self.global_step)

    def save_backward_activations(self):
        activations_norm = torch.norm(self.backwardOutput)
        self.writer.add_scalar(tag='{}/backward_activations'
                                   '_norm'.format(self.name),
                               scalar_value=activations_norm,
                               global_step=self.global_step)

    def save_backward_activations_hist(self):
        self.writer.add_histogram(tag='{}/backward_activations_'
                                      'hist'.format(
            self.name),
            values=self.backwardOutput,
            global_step=self.global_step)

    def save_state(self):
        """ Saves summary scalars (2-norm) of the gradients, weights and
         layer activations."""
        # Save norms
        self.save_activations()
        self.save_forward_weights()
        self.save_forward_weight_gradients()
        self.save_backward_weights()
        self.save_backward_activations()

    def save_state_histograms(self):
        """ The histograms (specified by the arguments) are saved to
        tensorboard"""
        # Save histograms
        self.save_forward_weights_gradients_hist()
        self.save_forward_weights_hist()
        self.save_activations_hist()
        self.save_backward_activations_hist()
        self.save_backward_weights_hist()

    def save_initial_backward_state(self):
        self.writer.add_histogram(tag='{}/backward_weights_initial_'
                                      'hist'.format(
            self.name),
            values=self.backwardWeights,
            global_step=0)
        self.writer.add_histogram(tag='{}/backward_bias_initial'
                                      'hist'.format(
            self.name),
            values=self.backwardBias,
            global_step=0)












