from create_datasets import GenerateDatasetFromModel
from optimizers import SGD, SGDMomentum
from neuralnetwork import Layer
from invertible_network import InvertibleInputLayer, \
    InvertibleLeakyReluLayer, InvertibleLinearOutputLayer, InvertibleNetwork
import torch
from tensorboard_api import Tensorboard
import utils.HelperFunctions as hf

# ======== set log directory ==========
log_dir = 'logs_hiddenlayer'
hf.init_logdir(log_dir)
Layer.tensorboard = Tensorboard(log_dir)


if torch.cuda.is_available():
    nb_gpus = torch.cuda.device_count()
    gpu_idx = nb_gpus - 1
    device = torch.device("cuda:{}".format(gpu_idx))
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('using GPU')
else:
    device = torch.device("cpu")
    print('using CPU')

# Create toy model dataset

input_layer_true = InvertibleInputLayer(layerDim=6,outDim=5,
                                        lossFunction='mse',
                                        name='input_layer_truemodel')
hidden_layer_true = InvertibleLeakyReluLayer(negativeSlope=0.01, inDim=6,
                                        layerDim=5, outDim=4,
                                             lossFunction='mse',
                                             name='hidden_layer_truemodel')
output_layer_true = InvertibleLinearOutputLayer(inDim=5, layerDim=4,
                                              stepsize=0.01,
                                                name='output_layer_truemodel')

true_network = InvertibleNetwork([input_layer_true,hidden_layer_true,
                                  output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(7000, 1)
input_dataset_test, output_dataset_test = generator.generate(1, 1000)



# Creating training network
inputlayer = InvertibleInputLayer(layerDim=6,outDim=5, lossFunction='mse',
                                  name='input_layer')
hiddenlayer = InvertibleLeakyReluLayer(negativeSlope=0.01, inDim=6,
                                        layerDim=5, outDim=4, lossFunction=
                                        'mse',
                                       name='hidden_layer')
outputlayer = InvertibleLinearOutputLayer(inDim=5, layerDim=4,
                                              stepsize=0.01,
                                          name='output_layer')

network = InvertibleNetwork([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer1 = SGD(network=network,threshold=0.0001, initLearningRate=0.1,
                 tau= 100,
                finalLearningRate=0.005, computeAccuracies= False, maxEpoch=120)
optimizer2 = SGDMomentum(network=network,threshold=1.2, initLearningRate=0.1,
                         tau=100, finalLearningRate=0.005,
                         computeAccuracies=False, maxEpoch=150, momentum=0.5)


# Train on dataset
optimizer1.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)


# ======== Run same experiment on shallow network ======
# creating separate logdir
log_dir = 'logs_shallow'
hf.init_logdir(log_dir)
Layer.tensorboard = Tensorboard(log_dir)

# Creating training network

inputlayer_shallow = InvertibleInputLayer(layerDim=6,outDim=4,
                                          lossFunction='mse',
                                          name='input_layer')

outputlayer_shallow = InvertibleLinearOutputLayer(inDim=6, layerDim=4,
                                              stepsize=0.01,
                                              name='output_layer')

network_shallow = InvertibleNetwork([inputlayer_shallow, outputlayer_shallow])
optimizer3 = SGD(network=network_shallow, threshold=0.0001,
                 initLearningRate=0.1,
                 tau= 100,
                finalLearningRate=0.005, computeAccuracies= False, maxEpoch=120)
optimizer3.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)

