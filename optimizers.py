import torch
import HelperFunctions as hf
from HelperClasses import NetworkError
import numpy as np
from neuralnetwork import Layer, Network

class Optimizer(object):
    """" Super class for all the different optimizers (e.g. SGD)"""

    def __init__(self, network, maxEpoch = 150, computeAccuracies = False):
        """
        :param network: network to train
        :param computeAccuracies: True if the optimizer should also save
        the accuracies. Only possible with
        classification problems
        :type network: Network
        """
        self.epoch = 0
        self.epochLosses = np.array([])
        self.batchLosses = np.array([])
        self.singleBatchLosses = np.array([])
        self.setNetwork(network)
        self.setComputeAccuracies(computeAccuracies)
        self.setMaxEpoch(maxEpoch)
        if self.computeAccuracies:
            self.epochAccuracies = np.array([])
            self.batchAccuracies = np.array([])
            self.singleBatchAccuracies = np.array([])

    def setNetwork(self,network):
        if not isinstance(network, Network):
            raise TypeError("Expecting Network object, instead got "
                            "{}".format(type(network)))
        self.network = network

    def setLearningRate(self, learningRate):
        if not isinstance(learningRate,float):
            raise TypeError("Expecting a float number as learningRate")
        if learningRate <=0:
            raise ValueError("Expecting a strictly positive learning rate")
        self.learningRate = learningRate

    def setThreshold(self, threshold):
        if not isinstance(threshold, float):
            raise TypeError("Expecting a float number as threshold")
        if threshold <=0:
            raise ValueError("Expecting a strictly positive threshold")
        self.threshold = threshold

    def setComputeAccuracies(self,computeAccuracies):
        if not isinstance(computeAccuracies,bool):
            raise TypeError("Expecting a bool as computeAccuracies")
        self.computeAccuracies = computeAccuracies

    def setMaxEpoch(self,maxEpoch):
        if not isinstance(maxEpoch,int):
            raise TypeError('Expecting integer for maxEpoch, got '
                            '{}'.format(type(maxEpoch)))
        if maxEpoch <= 0:
            raise ValueError('Expecting strictly positive integer for '
                             'maxEpoch, got {}'.format(maxEpoch))
        self.maxEpoch = maxEpoch


    def resetSingleBatchLosses(self):
        self.singleBatchLosses = np.array([])

    def resetSingleBatchAccuracies(self):
        self.singleBatchAccuracies = np.array([])

    def updateLearningRate(self):
        """ If the optimizer should do a specific update of the learningrate,
        this method should be overwritten in the subclass"""
        pass

    def saveResults(self, targets):
        """ Save the results of the optimizing step in the optimizer object."""
        self.batchLosses = np.append(self.batchLosses, self.network.loss(
            targets))
        self.singleBatchLosses = np.append(self.singleBatchLosses,
                                           self.network.loss(targets))
        if self.computeAccuracies:
            self.batchAccuracies = np.append(self.batchAccuracies,
                                             self.network.accuracy(targets))
            self.singleBatchAccuracies = np.append(
                self.singleBatchAccuracies, self.network.accuracy(targets))

    def runMNIST(self, trainLoader, device):
        """ Train the network on the total training set of MNIST as
        long as epoch loss is above the threshold
        :param trainLoader: a torch.utils.data.DataLoader object
        which containts the dataset"""
        if not isinstance(trainLoader, torch.utils.data.DataLoader):
            raise TypeError("Expecting a DataLoader object, now got a "
                            "{}".format(type(trainLoader)))

        epochLoss = float('inf')
        print('====== Training started =======')
        print('Epoch: ' + str(self.epoch) + ' ------------------------')
        while epochLoss > self.threshold and self.epoch < self.maxEpoch:
            for batch_idx, (data, target) in enumerate(trainLoader):
                if batch_idx % 50 == 0:
                    print('batch: ' + str(batch_idx))
                data = data.view(-1, 28*28, 1)
                target = hf.oneHot(target, 10)
                data, target = data.to(device), target.to(device)
                self.step(data, target)
            epochLoss = np.mean(self.singleBatchLosses)
            self.resetSingleBatchLosses()
            self.epochLosses = np.append(self.epochLosses, epochLoss)
            self.epoch += 1
            self.updateLearningRate()
            print('Loss: ' + str(epochLoss))
            if self.computeAccuracies:
                epochAccuracy = np.mean(self.singleBatchAccuracies)
                self.epochAccuracies = np.append(self.epochAccuracies,
                                                 epochAccuracy)
                self.resetSingleBatchAccuracies()
                print('Training Accuracy: ' + str(epochAccuracy))
            if self.epoch == self.maxEpoch:
                print('Training terminated, maximum epoch reached')
            print('Epoch: ' + str(self.epoch) + ' ------------------------')

        print('====== Training finished =======')

    def runDataset(self, inputData, targets):
        """ Train the network on a given dataset of size
         number of batches x batch size x input/target size x 1"""
        if not (inputData.size(0) == targets.size(0) and inputData.size(1) ==
                targets.size(1)):
            raise ValueError("InputData and Targets have not the same size")
        epochLoss = float('inf')
        print('====== Training started =======')
        print('Epoch: ' + str(self.epoch) + ' ------------------------')
        while epochLoss > self.threshold and self.epoch < self.maxEpoch:
            for i in range(inputData.size(0)):
                data = inputData[i,:,:,:]
                target = targets[i,:,:,:]
                if i % 500 == 0:
                    print('batch: ' + str(i))
                self.step(data, target)
            epochLoss = np.mean(self.singleBatchLosses)
            self.resetSingleBatchLosses()
            self.epochLosses = np.append(self.epochLosses, epochLoss)
            self.epoch += 1
            self.updateLearningRate()
            print('Loss: ' + str(epochLoss))
            if self.computeAccuracies:
                epochAccuracy = np.mean(self.singleBatchAccuracies)
                self.epochAccuracies = np.append(self.epochAccuracies,
                                                 epochAccuracy)
                self.resetSingleBatchAccuracies()
                print('Training Accuracy: ' + str(epochAccuracy))
            if self.epoch == self.maxEpoch:
                print('Training terminated, maximum epoch reached')
            print('Epoch: ' + str(self.epoch) + ' ------------------------')

        print('====== Training finished =======')


class SGD(Optimizer):
    """ Stochastic Gradient Descend optimizer"""

    def __init__(self, network, threshold, initLearningRate, tau=100,
                 finalLearningRate = None,
                 computeAccuracies = False, maxEpoch = 150):
        """
        :param threshold: the optimizer will run until the network loss is
        below this threshold
        :param initLearningRate: initial learning rate
        :param network: network to train
        :param computeAccuracies: True if the optimizer should also save the
        accuracies. Only possible with
        classification problems
        :param tau: used to update the learningrate according to
        learningrate = (1-epoch/tau)*initLearningRate +
                    epoch/tau* finalLearningRate
        :param finalLearningRate: see tau
        :type network: Network
        """
        super().__init__(network=network, maxEpoch=maxEpoch,
                         computeAccuracies=computeAccuracies)
        self.setThreshold(threshold)
        self.setLearningRate(initLearningRate)
        self.setInitLearningRate(initLearningRate)
        self.setTau(tau)
        if finalLearningRate is None:
            self.setFinalLearningRate(0.01*initLearningRate)
        else:
            self.setFinalLearningRate(finalLearningRate)

    def setInitLearningRate(self,initLearningRate):
        if not isinstance(initLearningRate,float):
            raise TypeError("Expecting float number for initLearningRate, got "
                            "{}".format(type(initLearningRate)))
        if initLearningRate <= 0:
            raise ValueError("Expecting strictly positive float, got "
                             "{}".format(initLearningRate))
        self.initLearningRate = initLearningRate

    def setTau(self,tau):
        if not isinstance(tau,int):
            raise TypeError("Expecting int number for tau, got"
                            " {}".format(type(tau)))
        if tau <= 0:
            raise ValueError("Expecting strictly positive integer, got "
                             "{}".format(tau))
        self.tau = tau

    def setFinalLearningRate(self,finalLearningRate):
        if not isinstance(finalLearningRate,float):
            raise TypeError("Expecting float number for finalLearningRate,"
                            " got {}".format(type(finalLearningRate)))
        if finalLearningRate <= 0:
            raise ValueError("Expecting strictly positive float, got "
                             "{}".format(finalLearningRate))
        self.finalLearningRate = finalLearningRate

    def updateLearningRate(self):
        if self.epoch <= self.tau:
            alpha = float(self.epoch)/float(self.tau)
            learningRate = (1. - alpha)*self.initLearningRate + \
                           alpha*self.finalLearningRate
            self.setLearningRate(learningRate)
        else:
            pass

    def step(self, inputBatch, targets):
        """ Perform one batch optimizing step"""
        self.network.propagateForward(inputBatch)
        self.network.propagateBackward(targets)
        self.network.computeGradients()
        self.network.updateParameters(self.learningRate)

        self.saveResults(targets)


class SGDMomentum(SGD):
    """ Stochastic Gradient Descend with momentum"""

    def __init__(self, network, threshold, initLearningRate, tau=100,
                 finalLearningRate=None,
                 computeAccuracies=False, maxEpoch=150, momentum = 0.5):
        """
        :param momentum: Momentum value that characterizes how much of the
        previous gradients is incorporated in the
                        update.
        """
        super().__init__(network=network, threshold=threshold,
                         initLearningRate=initLearningRate, tau=tau,
                         finalLearningRate=finalLearningRate,
                         computeAccuracies=computeAccuracies, maxEpoch=maxEpoch)
        self.setMomentum(momentum)
        self.network.initVelocities()

    def setMomentum(self, momentum):
        if not isinstance(momentum, float):
            raise TypeError("Expecting float number for momentum, "
                            "got {}".format(type(momentum)))
        if not (momentum>=0. and momentum < 1.):
            raise ValueError("Expecting momentum in [0;1), got {}".format(
                momentum))
        self.momentum = momentum

    def step(self, inputBatch, targets):
        """ Perform one batch optimizing step"""
        self.network.propagateForward(inputBatch)
        self.network.propagateBackward(targets)
        self.network.computeGradients()
        self.network.computeGradientVelocities(self.momentum,
                                                      self.learningRate)
        self.network.updateParametersWithVelocity()

        self.saveResults(targets)