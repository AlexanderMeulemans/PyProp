from utils.create_datasets import GenerateDatasetFromModel
from training.optimizers import SGD
from network_models.invertible_network import InvertibleInputLayer, \
InvertibleLeakyReluLayer, InvertibleLinearOutputLayer, InvertibleNetwork
from network_models.neuralnetwork import InputLayer, LeakyReluLayer, \
    LinearOutputLayer, Network
import torch
import numpy as np
import time
from tensorboardX import SummaryWriter
from utils.LLS import linear_least_squares

# ======== User variables ============
training_size = 1000
testing_size = 100

# ======== set log directory ==========
log_dir = '../logs/toyexample_BP'
writer = SummaryWriter(log_dir=log_dir)


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

input_layer_true = InputLayer(layerDim=6, writer=writer,
                              name='input_layer_true_model')
hidden_layer_true = LeakyReluLayer(negativeSlope=0.1,inDim=6,layerDim=10,
                                   writer=writer,
                                   name='hidden_layer_true_model')
output_layer_true = LinearOutputLayer(inDim=5, layerDim=4,
                                      lossFunction='mse',
                                      writer=writer,
                                      name='output_layer_true_model')
true_network = Network([input_layer_true, hidden_layer_true,
                                  output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(training_size, 10)
input_dataset_test, output_dataset_test = generator.generate(
    testing_size, 10)

# compute least squares solution as control
print('computing LS solution ...')
weights, train_loss, test_loss = linear_least_squares(input_dataset,
                                                      output_dataset,
                                                      input_dataset_test,
                                                      output_dataset_test)
print('LS train loss: '+str(train_loss))
print('LS test loss: '+str(test_loss))

# ===== Run experiment with invertible TP =======

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
                finalLearningRate=0.005, computeAccuracies= False, maxEpoch=120,
                 outputfile_name='resultfile.csv')



# Train on dataset
timings = np.array([])
start_time = time.time()
optimizer1.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time-start_time))
timings = np.append(timings, end_time-start_time)