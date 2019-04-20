# sacramento@ini.ethz.ch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import joao.bionets as bionets

# MODEL = 'FA'
# CONFIG_FILENAME = 'etc/mnist_feedback_alignment_config.json'

#MODEL = 'TP'
#CONFIG_FILENAME = 'etc/mnist_tp_config.json'

MODEL = 'CAPSULE'
CONFIG_FILENAME = 'etc/mnist_capsule_config.json'


# Used only to measure performance.
# Changing this variable doesn't affect the outcome of learning.
loss_function = nn.CrossEntropyLoss()

def train(config, log_file, model, device, train_loader, epoch):
    print('start training')
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        model.learn(config, target)
        if batch_idx % config['log-interval'] == 0:
            loss = model.loss_function(target)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss), file=log_file)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))
            log_file.flush()

def test(config, log_file, model, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += model.loss_function(target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), file=log_file)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    log_file.flush()


with open(CONFIG_FILENAME) as mnist_config_json_file:
    config = json.load(mnist_config_json_file)

log_file = open(config['log-file'], 'w')
use_cuda = config['cuda'] and torch.cuda.is_available()

if config['seed'] is not None:
    torch.manual_seed(config['seed'])

device = torch.device("cuda" if use_cuda else "cpu")
config['device'] = device

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../experiments/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=config['batch-size'], shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../experiments/data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=config['test-batch-size'], shuffle=True, **kwargs)

if MODEL == 'FA':
    model = bionets.FANet(config)
elif MODEL == 'TP':
    model = bionets.TPNet(config)
elif MODEL == 'CAPSULE':
    model = bionets.CapsuleNetBP(config)
parameters_init = model.get_numpy_parameters()


print('Training model...')
for epoch in range(1, config['epochs'] + 1):
    train(config, log_file, model, device, train_loader, epoch)
    test(config, log_file, model, device, test_loader)
print('Done.')

parameters_end = model.get_numpy_parameters()
