import sys
sys.path.append(r"c:\users\alexander\appdata\local\programs\python\python36\lib\site-packages")

from neuralnetwork import ReluLayer, InputLayer, SoftmaxOutputLayer, Network
from optimizers import SGD, SGDMomentum
import torch
import torchvision
from utils import HelperFunctions as hf

# Initializing network

if torch.cuda.is_available():
    nb_gpus = torch.cuda.device_count()
    gpu_idx = nb_gpus - 1
    device = torch.device("cuda:{}".format(gpu_idx))
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('using GPU')
else:
    device = torch.device("cpu")
    print('using CPU')
inputlayer = InputLayer(28*28)
hiddenlayer = ReluLayer(28*28,100)
outputlayer = SoftmaxOutputLayer(100,10,'crossEntropy')

network = Network([inputlayer, hiddenlayer, outputlayer])
if torch.cuda.is_available():
    network.cuda(torch.cuda.current_device())

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

batch_size = 128

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

