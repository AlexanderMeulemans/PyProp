"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np


def linear_least_squares(input_dataset, output_dataset, input_dataset_test,
                         output_dataset_test):
    # compute least squares solution as control
    print('computing LS solution ...')
    input_dataset_np = input_dataset.cpu().numpy()
    output_dataset_np = output_dataset.cpu().numpy()
    input_dataset_test_np = input_dataset_test.cpu().numpy()
    output_dataset_test_np = output_dataset_test.cpu().numpy()

    input_dataset_np = np.reshape(input_dataset_np,
                                  (input_dataset_np.shape[0] * \
                                   input_dataset_np.shape[1],
                                   input_dataset_np.shape[2]))
    output_dataset_np = np.reshape(output_dataset_np,
                                   (output_dataset_np.shape[0] * \
                                    output_dataset_np.shape[1],
                                    output_dataset_np.shape[2]))
    input_dataset_test_np = np.reshape(input_dataset_test_np,
                                       (input_dataset_test_np.shape[0] * \
                                        input_dataset_test_np.shape[1],
                                        input_dataset_test_np.shape[2]))
    output_dataset_test_np = np.reshape(output_dataset_test_np,
                                        (output_dataset_test_np.shape[0] * \
                                         output_dataset_test_np.shape[1],
                                         output_dataset_test_np.shape[2]))

    weights = np.linalg.lstsq(input_dataset_np, output_dataset_np)
    weights = weights[0]
    train_loss = np.mean(np.sum(np.square(np.matmul(input_dataset_np, weights) \
                                          - output_dataset_np), axis=1))
    test_loss = np.mean(np.sum(np.square(np.matmul(input_dataset_test_np,
                                                   weights) \
                                         - output_dataset_test_np), axis=1))
    return weights, train_loss, test_loss
