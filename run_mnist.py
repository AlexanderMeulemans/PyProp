import sys
sys.path.append(r"c:\users\alexander\appdata\local\programs\python\python36\lib\site-packages")

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from Backprop import Layer, Network
import HelperFunctions as hf
import numpy as np



train_set = torchvision.datasets.MNIST(root='./data', train = True, download=True,
                                transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

batch_size = 128

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1000,
                shuffle=False)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# create network:
learningRate = 0.1
network = Network([784,300,100,10],learningRate=learningRate)

# Train network:

print('====== Training started ======')
epoch_error = float('inf')
epoch_error_array = np.array([])
training_accuracy_array = np.array([])
threshold = 0.1
epoch = 0
while epoch_error > threshold:
    for batch_idx, (data,target) in enumerate(train_loader):
        if batch_idx %10 == 0:
            print('batch: ' + str(batch_idx))
        data = data.view(-1,28*28,1)
        target = hf.oneHot(target,10)
        network.batchTraining(data,target)
    epoch_error = torch.mean(network.loss).numpy()
    epoch_error_array = np.append(epoch_error_array, epoch_error)
    training_accuracy = torch.mean(network.accuracyLst).numpy()
    training_accuracy_array = np.append(training_accuracy_array, training_accuracy)
    network.resetLoss()
    epoch +=1
    print('Epoch: ' + str(epoch) + ' ------------------------')
    print('Loss: ' + str(epoch_error))
    print('Training Accuracy: ' + str(training_accuracy))

print('====== Training finished =======')

# test network
for batch_idx, (data,target) in enumerate(test_loader):
    data = data.view(-1, 28 * 28, 1)
    target = hf.oneHot(target, 10)
    test_loss = network.test_loss(data,target)
    test_accuracy = network.accuracy(data,target)
    print('Test Loss: ' + str(test_loss))
    print('Test Accuracy: ' + str(test_accuracy))

# plot training process
hf.plot_epochs(epoch_error_array,training_accuracy)













# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig

