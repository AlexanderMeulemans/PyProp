import torch
from utils import HelperFunctions as hf
from networks.network import Network
import pandas as pd


class Optimizer(object):
    """" Super class for all the different optimizers (e.g. SGD)"""

    def __init__(self, network, maxEpoch = 150, computeAccuracies = False,
                 outputfile_name = 'result_file.csv'):
        """
        :param network: network to train
        :param computeAccuracies: True if the optimizer should also save
        the accuracies. Only possible with
        classification problems
        :type network: Network
        """
        self.epoch = 0
        self.epochLosses = torch.Tensor([])
        self.batchLosses = torch.Tensor([])
        self.singleBatchLosses = torch.Tensor([])
        self.testLosses = torch.Tensor([])
        self.testBatchLosses = torch.Tensor([])
        self.setNetwork(network)
        self.setComputeAccuracies(computeAccuracies)
        self.setMaxEpoch(maxEpoch)
        self.writer = self.network.writer
        self.global_step = 0
        self.outputfile = pd.DataFrame(columns=['Train_loss', 'Test_loss'])
        self.outputfile_name = '../logs/{}'.format(outputfile_name)
        if self.computeAccuracies:
            self.epochAccuracies = torch.Tensor([])
            self.batchAccuracies = torch.Tensor([])
            self.singleBatchAccuracies = torch.Tensor([])
            self.testAccuracies = torch.Tensor([])
            self.testBatchAccuracies = torch.Tensor([])
            self.outputfile = pd.DataFrame(columns=
                                           ['Train_loss', 'Test_loss',
                                            'Train_accuracy', 'Test_accuracy'])

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
        self.singleBatchLosses = torch.Tensor([])

    def resetSingleBatchAccuracies(self):
        self.singleBatchAccuracies = torch.Tensor([])

    def resetTestBatchLosses(self):
        self.testBatchLosses = torch.Tensor([])

    def resetTestBatchAccuracies(self):
        self.testBatchAccuracies = torch.Tensor([])

    def resetOptimizer(self):
        self.epoch = 0
        self.epochLosses = torch.Tensor([])
        self.batchLosses = torch.Tensor([])
        self.singleBatchLosses = torch.Tensor([])
        self.testLosses = torch.Tensor([])
        self.testBatchLosses = torch.Tensor([])
        self.global_step = 0
        if self.computeAccuracies:
            self.epochAccuracies = torch.Tensor([])
            self.batchAccuracies = torch.Tensor([])
            self.singleBatchAccuracies = torch.Tensor([])
            self.testAccuracies = torch.Tensor([])
            self.testBatchAccuracies = torch.Tensor([])

    def updateLearningRate(self):
        """ If the optimizer should do a specific update of the learningrate,
        this method should be overwritten in the subclass"""
        pass

    def saveResults(self, targets):
        """ Save the results of the optimizing step in the optimizer object."""
        loss = self.network.loss(targets)
        self.writer.add_scalar(tag='training_loss_batch',
                               scalar_value=loss,
                               global_step=self.global_step)
        self.batchLosses = torch.cat([self.batchLosses, loss])
        self.singleBatchLosses = torch.cat([self.singleBatchLosses, loss])
        self.network.save_state(self.global_step)
        if self.computeAccuracies:
            accuracy = self.network.accuracy(targets)
            self.writer.add_scalar(tag='training_accuracy_batch',
                                   scalar_value=accuracy,
                                   global_step=self.global_step)
            self.batchAccuracies = torch.cat([self.batchAccuracies, accuracy],0)
            self.singleBatchAccuracies = torch.cat(
                [self.singleBatchAccuracies, accuracy],0)

    def test_step(self, data, target):
        self.network.propagateForward(data)
        self.save_test_results_batch(target)

    def save_test_results_batch(self, target):
        batch_loss = self.network.loss(target)
        self.testBatchLosses = torch.cat([self.testBatchLosses, batch_loss], 0)
        if self.computeAccuracies:
            batch_accuracy = self.network.accuracy(target)
            self.testBatchAccuracies = torch.cat([self.testBatchAccuracies,
                                                 batch_accuracy])

    def save_test_results_epoch(self):
        test_loss = torch.Tensor([torch.mean(self.testBatchLosses)])
        self.testLosses = torch.cat([self.testLosses, test_loss], 0)
        self.writer.add_scalar(tag='test_loss',
                               scalar_value=test_loss,
                               global_step=self.epoch)
        self.resetTestBatchLosses()
        print('Test Loss: ' + str(test_loss))
        if self.computeAccuracies:
            test_accuracy = torch.Tensor([torch.mean(self.testBatchAccuracies)])
            self.testAccuracies = torch.cat([self.testAccuracies, test_accuracy]
                                            , 0)
            self.writer.add_scalar(tag='test_accuracy',
                                   scalar_value=test_accuracy,
                                   global_step=self.epoch)
            self.resetTestBatchAccuracies()
            print('Test Accuracy: ' + str(test_accuracy))

    def save_train_results_epoch(self):
        epochLoss = torch.Tensor([torch.mean(self.singleBatchLosses)])
        self.writer.add_scalar(tag='train_loss', scalar_value=epochLoss,
                               global_step=self.epoch)
        self.resetSingleBatchLosses()
        self.epochLosses = torch.cat([self.epochLosses, epochLoss], 0)
        self.network.save_state_histograms(self.epoch)
        print('Train Loss: ' + str(epochLoss))
        if self.computeAccuracies:
            epochAccuracy = torch.Tensor([torch.mean(self.singleBatchAccuracies)
                                          ])
            self.epochAccuracies = torch.cat([self.epochAccuracies,
                                             epochAccuracy], 0)
            self.writer.add_scalar(tag='train_accuracy',
                                   scalar_value=epochAccuracy,
                                   global_step=self.epoch)
            self.resetSingleBatchAccuracies()
            print('Train Accuracy: ' + str(epochAccuracy))

    def save_result_file(self):
        train_loss = self.epochLosses[-1]
        test_loss = self.testLosses[-1]
        if self.computeAccuracies:
            train_accuracy = self.epochAccuracies[-1]
            test_accuracy = self.testAccuracies[-1]
            self.outputfile.loc[self.epoch] = [train_loss, test_loss,
                                               train_accuracy, test_accuracy]
        else:
            self.outputfile.loc[self.epoch] = [train_loss, test_loss]

    def runMNIST(self, trainLoader, testLoader, device):
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
                if batch_idx % 5 == 0:
                    print('batch: ' + str(batch_idx))
                data = data.view(-1, 28*28, 1)
                target = hf.oneHot(target, 10)
                data, target = data.to(device), target.to(device)
                self.step(data, target)
            self.save_train_results_epoch()
            epochLoss = self.epochLosses[-1]
            self.testMNIST(testLoader, device)
            self.save_result_file()
            self.epoch += 1
            self.updateLearningRate()
            if self.epoch == self.maxEpoch:
                print('Training terminated, maximum epoch reached')
            print('Epoch: ' + str(self.epoch) + ' ------------------------')
        self.global_step = 0
        self.save_csv_file()
        self.writer.close()
        print('====== Training finished =======')

    def testMNIST(self, testLoader, device):
        for batch_idx, (data,target) in enumerate(testLoader):
            data = data.view(-1, 28 * 28, 1)
            target = hf.oneHot(target, 10)
            data, target = data.to(device), target.to(device)
            self.test_step(data, target)
        self.save_test_results_epoch()

    def runDataset(self, inputData, targets, inputDataTest, targetsTest):
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
                if i % 100 == 0:
                    print('batch: ' + str(i))
                self.step(data, target)
            self.save_train_results_epoch()
            epochLoss = self.epochLosses[-1]
            self.testDataset(inputDataTest, targetsTest)
            self.save_result_file()
            self.epoch += 1
            self.updateLearningRate()
            if self.epoch == self.maxEpoch:
                print('Training terminated, maximum epoch reached')
            print('Epoch: ' + str(self.epoch) + ' ------------------------')

        self.global_step = 0
        self.save_csv_file()
        self.writer.close()
        print('====== Training finished =======')

    def testDataset(self, inputData, targets):
        for i in range(inputData.size(0)):
            data = inputData[i, :, :, :]
            target = targets[i, :, :, :]
            self.test_step(data, target)
        self.save_test_results_epoch()

    def step(self, inputBatch, targets):
        raise NotImplementedError

    def save_csv_file(self):
        self.outputfile.to_csv(self.outputfile_name)


class SGD(Optimizer):
    """ Stochastic Gradient Descend optimizer"""

    def __init__(self, network, threshold, initLearningRate, tau=100,
                 finalLearningRate = None,
                 computeAccuracies = False, maxEpoch = 150,
                 outputfile_name='resultfile.csv'):
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
                         computeAccuracies=computeAccuracies,
                         outputfile_name=outputfile_name)
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
        self.global_step += 1


class SGDMomentum(SGD):
    """ Stochastic Gradient Descend with momentum"""

    def __init__(self, network, threshold, initLearningRate, tau=100,
                 finalLearningRate=None,
                 computeAccuracies=False, maxEpoch=150, momentum = 0.5,
                 outputfile_name = 'resultfile.csv'):
        """
        :param momentum: Momentum value that characterizes how much of the
        previous gradients is incorporated in the
                        update.
        """
        super().__init__(network=network, threshold=threshold,
                         initLearningRate=initLearningRate, tau=tau,
                         finalLearningRate=finalLearningRate,
                         computeAccuracies=computeAccuracies, maxEpoch=maxEpoch,
                         outputfile_name=outputfile_name)
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
        self.global_step += 1