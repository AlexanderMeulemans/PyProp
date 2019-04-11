"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
import matplotlib.pyplot as plt
import os


def kronecker(i, j):
    if i == j:
        return 1.0
    else:
        return 0.0


def one_hot(targets, nb_of_classes):
    batch_size = len(targets)
    one_hot_targets = torch.zeros(batch_size, nb_of_classes, 1)
    for i in range(batch_size):
        one_hot_targets[i, targets[i], 0] = 1.0
    return one_hot_targets


def plot_epochs(loss, accuracy, gradients=None):
    epochs = range(len(loss))
    plt.figure()
    if gradients == None:
        plt.subplot(211)
        plt.plot(epochs, loss)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.subplot(212)
        plt.plot(epochs, accuracy)
        plt.title('Accuracy')
        plt.xlabel('Epoch')

    else:
        plt.subplot(311)
        plt.plot(epochs, loss)
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
    """ Convert the class probabilities to one
    predicted class per batch sample"""
    probabilities = torch.reshape(probabilities, (probabilities.shape[0],
                                                  probabilities.shape[1]))
    return torch.argmax(probabilities, dim=1)


def accuracy(predictions, targets):
    """ Return the accuracy of the batch"""
    if not predictions.size() == targets.size():
        raise ValueError(
            "Expecting equal dimension for predictions and targets")

    total_labels = float(predictions.size(0))
    correct_labels = 0.
    for i in range(predictions.size(0)):
        if predictions[i] == targets[i]:
            correct_labels += 1.
    return torch.Tensor([correct_labels / total_labels])


def contains_no_nans(tensor):
    a = tensor == tensor
    return a.sum().numpy() == a.view(-1).size(0)


def contains_nans(tensor):
    a = tensor != tensor
    return a.any()


def init_logdir(dr):
    directory = os.path.join(os.path.curdir, dr)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_invertible_random_matrix(rows, cols, threshold=0.4, max_iter=300):
    cpu = torch.device('cpu')
    m = torch.randn(rows, cols)
    device = m.device
    m = m.to(cpu)
    U, S, V = torch.svd(m)
    s_max = S[0]
    s_min = S[-1]
    iter = 0

    while s_max * s_min < threshold:
        m = torch.randn(rows, cols)
        m = m.to(cpu)
        U, S, V = torch.svd(m)
        s_max = S[0]
        s_min = S[-1]
        iter += 1
        if iter >= max_iter:
            raise RuntimeWarning('max iterations reached of '
                                 'get_invertible_random_matrix.'
                                 'random matrix returned without '
                                 'checking the singular values')

    return m.to(device)

def get_invertible_neighbourhood_matrix(matrix, distance, threshold=0.4):
    cpu = torch.device('cpu')
    m = torch.randn(matrix.shape[0], matrix.shape[1])
    device = m.device
    m = m.to(cpu)
    matrix = matrix.to(cpu)
    norm = torch.norm(m)
    m = distance/norm*m
    matrix_n = matrix + m
    U, S, V = torch.svd(matrix_n)
    s_max = S[0]
    s_min = S[-1]
    iter = 0
    while s_max * s_min < threshold:
        m = torch.randn(matrix.shape[0], matrix.shape[1])
        m.to(cpu)
        norm = torch.norm(m)
        m = distance / norm * m
        matrix_n = matrix + m
        U, S, V = torch.svd(matrix_n)
        s_max = S[0]
        s_min = S[-1]
        iter += 1

    return matrix_n.to(device)