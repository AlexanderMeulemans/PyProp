from utils.create_datasets import GenerateDatasetFromModel
from optimizers.optimizers import SGD, SGDMomentum
from layers.invertible_layer import InvertibleInputLayer, \
    InvertibleLeakyReluLayer, InvertibleSoftmaxOutputLayer, InvertibleNetwork
import torch

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('using GPU')
else:
    print('using CPU')

# Create toy model dataset

input_layer_true = InvertibleInputLayer(layerDim=6,outDim=5,
                                        lossFunction='mse')
hidden_layer_true = InvertibleLeakyReluLayer(negativeSlope=0.01, inDim=6,
                                        layerDim=5, outDim=4, lossFunction=
                                        'mse')
output_layer_true = InvertibleSoftmaxOutputLayer(inDim=5, layerDim=4,
                                              stepsize=0.05)

true_network = InvertibleNetwork([input_layer_true,hidden_layer_true,
                                  output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(7000, 1)
input_dataset_test, output_dataset_test = generator.generate(1000, 1)



# Creating training network
inputlayer = InvertibleInputLayer(layerDim=6,outDim=5, lossFunction='mse')
hiddenlayer = InvertibleLeakyReluLayer(negativeSlope=0.01, inDim=6,
                                        layerDim=5, outDim=4, lossFunction=
                                        'mse')
outputlayer = InvertibleSoftmaxOutputLayer(inDim=5, layerDim=4,
                                              stepsize=0.05)

network = InvertibleNetwork([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer1 = SGD(network=network,threshold=0.01, initLearningRate=0.01,
                 tau= 100,
                finalLearningRate=0.0005, computeAccuracies= False, maxEpoch=120)
optimizer2 = SGDMomentum(network=network,threshold=1.2, initLearningRate=0.1,
                         tau=100, finalLearningRate=0.005,
                         computeAccuracies=False, maxEpoch=150, momentum=0.5)


# Train on MNIST
optimizer1.runDataset(input_dataset, output_dataset)

# Test network

predicted_classes = network.predict(input_dataset_test[0,:,:,:])
test_loss = network.loss(output_dataset_test[0,:,:,:])
print('Test Loss: ' + str(test_loss))