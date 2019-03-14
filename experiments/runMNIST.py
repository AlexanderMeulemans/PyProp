from layers.layer import ReluLayer, InputLayer, SoftmaxOutputLayer
from layers.network import Network
from optimizers.optimizers import SGD, SGDMomentum
import torch
import torchvision
from tensorboardX import SummaryWriter
import os

# Initializing network

# ======== set log directory ==========
log_dir = '../logs/MNIST_BP_500n'
writer = SummaryWriter(log_dir=log_dir)

# ======== set device ============
if torch.cuda.is_available():
    gpu_idx = 0
    device = torch.device("cuda:{}".format(gpu_idx))
    # IMPORTANT: set_default_tensor_type uses automatically device 0,
    # untill now, I did not find a fix for this
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('using GPU')
else:
    device = torch.device("cpu")
    print('using CPU')

# ======== Design network =============

inputlayer = InputLayer(layerDim=28*28, writer=writer, name='input_layer_BP')
hiddenlayer = ReluLayer(inDim=28*28, layerDim=500, writer=writer,
                                       name='hidden_layer_BP')
outputlayer = SoftmaxOutputLayer(inDim=500, layerDim=10,
                                 lossFunction='crossEntropy',
                                 name='output_layer_BP',
                                 writer=writer)

network = Network([inputlayer, hiddenlayer, outputlayer])
# if torch.cuda.is_available():
#     network.cuda(device)

# Loading dataset
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

batch_size = 1024

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1000,
                shuffle=False)

# Initializing optimizer
optimizer1 = SGD(network=network,threshold=1.2, initLearningRate=0.1, tau= 100,
                finalLearningRate=0.005, computeAccuracies= True, maxEpoch=120)
optimizer2 = SGDMomentum(network=network,threshold=1.2, initLearningRate=0.1,
                         tau=100, finalLearningRate=0.005,
                         computeAccuracies=True, maxEpoch=150, momentum=0.5)


# Train on MNIST
optimizer1.runMNIST(train_loader, test_loader, device)

# Test network
# for batch_idx, (data,target) in enumerate(test_loader):
#     data = data.view(-1, 28 * 28, 1)
#     target = hf.oneHot(target, 10)
#     data, target = data.to(device), target.to(device)
#     predicted_classes = network.predict(data)
#     test_loss = network.loss(target)
#     test_accuracy = network.accuracy(target)
#     print('Test Loss: ' + str(test_loss))
#     print('Test Accuracy: ' + str(test_accuracy))

