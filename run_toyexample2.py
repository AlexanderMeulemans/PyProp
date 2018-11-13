from create_datasets import GenerateDatasetFromModel
from optimizers import SGD, SGDMomentum
from targetprop_network import InvertibleInputLayer, \
    InvertibleLeakyReluLayer, InvertibleLinearOutputLayer, InvertibleNetwork

# Create toy model dataset

input_layer_true = InvertibleInputLayer(layerDim=6,outDim=5,
                                        lossFunction='mse')
hidden_layer_true = InvertibleLeakyReluLayer(negativeSlope=0.01, inDim=6,
                                        layerDim=5, outDim=4, lossFunction=
                                        'mse')
output_layer_true = InvertibleLinearOutputLayer(inDim=5, layerDim=4,
                                              stepsize=0.01)

true_network = InvertibleNetwork([input_layer_true,hidden_layer_true,
                                  output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(700, 128)
input_dataset_test, output_dataset_test = generator.generate(1, 1000)



# Creating training network
inputlayer = InvertibleInputLayer(layerDim=6,outDim=5, lossFunction='mse')
hiddenlayer = InvertibleLeakyReluLayer(negativeSlope=0.01, inDim=6,
                                        layerDim=5, outDim=4, lossFunction=
                                        'mse')
outputlayer = InvertibleLinearOutputLayer(inDim=5, layerDim=4,
                                              stepsize=0.01)

network = InvertibleNetwork([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer1 = SGD(network=network,threshold=0.01, initLearningRate=0.1,
                 tau= 100,
                finalLearningRate=0.005, computeAccuracies= False, maxEpoch=120)
optimizer2 = SGDMomentum(network=network,threshold=1.2, initLearningRate=0.1,
                         tau=100, finalLearningRate=0.005,
                         computeAccuracies=False, maxEpoch=150, momentum=0.5)


# Train on MNIST
optimizer1.runDataset(input_dataset, output_dataset)

# Test network

predicted_classes = network.predict(input_dataset_test[0,:,:,:])
test_loss = network.loss(output_dataset_test[0,:,:,:])
print('Test Loss: ' + str(test_loss))