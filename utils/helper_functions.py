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
import numpy as np
import pandas as pd


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

def get_angle(tensor1, tensor2):
    """
    returns angle between each sample of the two batches (tensors of size
    batch_size x vector dimension x 1
    """

    tensor1_T = torch.transpose(tensor1,1,2)
    tensor2_T = torch.transpose(tensor2, 1, 2)
    inner_product = torch.bmm(tensor1_T,tensor2)
    norm_1 = torch.sqrt(torch.bmm(tensor1_T, tensor1))
    norm_2 = torch.sqrt(torch.bmm(tensor2_T, tensor2))
    angles = inner_product/(norm_1*norm_2)
    return angles.squeeze()

def append_results(array, max_len):
    """
    Appends the given array by its last element until max_len is reached.
    """
    if len(array) == max_len:
        return array
    else:
        last_element = array[-1]
        for i in range(max_len-len(array)):
            array = np.append(array, last_element)
        return array

def get_invertible_diagonal_matrix(size, threshold=0.1, max_iter=500):
    diag = torch.randn((size))
    while torch.min(torch.abs(diag)) < threshold:
        diag = torch.randn((size))
    return torch.diag(diag)

def eye(rows, cols=None, batch_size=1):
    if cols is None:
        cols = rows
    output = torch.empty(batch_size, rows, cols)
    for i in range(batch_size):
        output[i,:,:] = torch.eye(rows, cols)
    return output

def pinverse(tensor, rcond=1e-6):
    output = torch.empty((tensor.shape[0], tensor.shape[2], tensor.shape[1]))
    for i in range(tensor.shape[0]):
        output[i,:,:] = torch.pinverse(tensor[i,:,:], rcond=rcond)
    return output

def get_stats_gridsearch(results, distances, learning_rates):
    best_results = np.min(results, 2)
    succesful_runs = best_results != 0
    descending_runs = np.zeros(succesful_runs.shape,dtype=bool)
    for i in range(len(distances)):
        for j in range(len(learning_rates)):
            if succesful_runs[i,j]:
                if is_descending_run(results[i,j,:]):
                    descending_runs[i,j] = True
    best_results_distance = np.zeros(len(distances))
    best_learning_rates = np.zeros(len(distances))
    success_counts = np.zeros(len(distances))
    descending_counts = np.zeros(len(distances))
    for i in range(len(distances)):
        valid_run = False
        best_results_distance[i] = float('inf')
        success_counts[i] = np.sum(succesful_runs[i,:])
        descending_counts[i] = np.sum(descending_runs[i,:])
        for j, learning_rate in enumerate(learning_rates):
            if descending_runs[i,j]:
                valid_run = True
                best_result = best_results[i,j]
                if best_result < best_results_distance[i]:
                    best_results_distance[i] = best_result
                    best_learning_rates[i] = learning_rate
        if not valid_run:
            best_results_distance[i] = np.nan
            best_learning_rates[i] = np.nan
    result_array = np.empty((len(distances), 4))
    result_array[:,0] = success_counts
    result_array[:,1] = descending_counts
    result_array[:,2] = best_results_distance
    result_array[:,3] = best_learning_rates
    columns = ['success_count', 'descending_count', 'best_result', 'best_learning_rate']
    result_frame = pd.DataFrame(result_array,index=distances,columns=columns)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(result_frame)
    return result_frame

def get_stats_gridsearch2(results, distances, learning_rates, weight_decays):
    best_results = np.min(results, 3)
    succesful_runs = best_results != 0
    descending_runs = np.zeros(succesful_runs.shape,dtype=bool)
    for i in range(len(distances)):
        for j in range(len(learning_rates)):
            for k in range(len(weight_decays)):
                if succesful_runs[i,j,k]:
                    if is_descending_run(results[i,j,k,:]):
                        descending_runs[i,j,k] = True
    best_results_distance = np.zeros(len(distances))
    best_learning_rates = np.zeros(len(distances))
    best_weight_decays = np.zeros(len(distances))
    success_counts = np.zeros(len(distances))
    descending_counts = np.zeros(len(distances))
    for i in range(len(distances)):
        valid_run = False
        best_results_distance[i] = float('inf')
        success_counts[i] = np.sum(succesful_runs[i,:,:])
        descending_counts[i] = np.sum(descending_runs[i,:,:])
        for j, learning_rate in enumerate(learning_rates):
            for k, weight_decay in enumerate(weight_decays):
                if descending_runs[i,j,k]:
                    valid_run = True
                    best_result = best_results[i,j,k]
                    if best_result < best_results_distance[i]:
                        best_results_distance[i] = best_result
                        best_learning_rates[i] = learning_rate
                        best_weight_decays[i] = weight_decay
        if not valid_run:
            best_results_distance[i] = np.nan
            best_learning_rates[i] = np.nan
            best_weight_decays[i] = np.nan
    result_array = np.empty((len(distances), 5))
    result_array[:,0] = success_counts
    result_array[:,1] = descending_counts
    result_array[:,2] = best_results_distance
    result_array[:,3] = best_learning_rates
    result_array[:,4] = best_weight_decays
    columns = ['success_count', 'descending_count', 'best_result',
               'best_learning_rate', 'best_weight_decay']
    result_frame = pd.DataFrame(result_array, index=distances, columns=columns)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(result_frame)
    return result_frame




def is_descending_run(loss_array, threshold=1.5):
    min_loss = np.min(loss_array)
    if loss_array[-1] < threshold*min_loss:
        return True
    else:
        return False
