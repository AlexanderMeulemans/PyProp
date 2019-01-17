import torch
import matplotlib.pyplot as plt
import os

def kronecker(i,j):
    if i == j:
        return 1.0
    else:
        return 0.0

def oneHot(targets,nbOfClasses):
    batchSize = len(targets)
    oneHotTargets = torch.zeros(batchSize,nbOfClasses,1)
    for i in range(batchSize):
        oneHotTargets[i,targets[i],0] = 1.0
    return oneHotTargets

def plot_epochs(loss,accuracy, gradients = None):
    epochs = range(len(loss))
    plt.figure()
    if gradients == None:
        plt.subplot(211)
        plt.plot(epochs,loss)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.subplot(212)
        plt.plot(epochs,accuracy)
        plt.title('Accuracy')
        plt.xlabel('Epoch')

    else:
        plt.subplot(311)
        plt.plot(epochs,loss)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.subplot(312)
        plt.plot(epochs, accuracy)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.subplot(313)
        plt.plot(epochs, gradients)
        plt.title('Gradients')
        plt.xlabel('Epoch')

def prob2class(probabilities):
    """ Convert the class probabilities to one predicted class per batch sample"""
    probabilities = torch.reshape(probabilities, (probabilities.shape[0],
                                                  probabilities.shape[1]))
    return torch.argmax(probabilities, dim=1)

def accuracy(predictions, targets):
    """ Return the accuracy of the batch"""
    if not predictions.size() == targets.size():
        raise ValueError("Expecting equal dimension for predictions and targets")

    total_labels = float(predictions.size(0))
    correct_labels = 0.
    for i in range(predictions.size(0)):
        if predictions[i] == targets[i]:
            correct_labels += 1.
    return correct_labels/total_labels

def containsNoNaNs(tensor):
    a = tensor == tensor
    return a.sum().numpy()==a.view(-1).size(0)

def containsNaNs(tensor):
    a = tensor != tensor
    return a.any()

def init_logdir(dr):
    directory = os.path.join(os.path.curdir, dr)
    if not os.path.exists(directory):
        os.makedirs(directory)
