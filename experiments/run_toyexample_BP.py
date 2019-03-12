from utils.create_datasets import GenerateDatasetFromModel
from training.optimizers import SGD
from network_models.neuralnetwork import InputLayer, ReluLayer, \
    LinearOutputLayer, Network
import torch
import time
from tensorboardX import SummaryWriter
from utils.LLS import linear_least_squares

torch.manual_seed(47)

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
hidden_layer_true = ReluLayer(inDim=6,layerDim=10, writer=writer,
                              name='hidden_layer_true_model')
output_layer_true = LinearOutputLayer(inDim=10, layerDim=4,
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


# ===== Run same experiment with backprop =======

# Creating training network
inputlayer = InputLayer(layerDim=6, writer=writer, name='input_layer_BP')
hiddenlayer = ReluLayer(inDim=6, layerDim=10, writer=writer,
                                       name='hidden_layer_BP')
outputlayer = LinearOutputLayer(inDim=10, layerDim=4, lossFunction='mse',
                                          name='output_layer_BP',
                                writer=writer)

network_backprop = Network([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer4 = SGD(network=network_backprop, threshold=0.0001,
                 initLearningRate=0.1,
                 tau=100,
                 finalLearningRate=0.005, computeAccuracies=False,
                 maxEpoch=5,
                 outputfile_name='resultfile_BP.csv')



# Train on dataset
start_time = time.time()
optimizer4.runDataset(input_dataset, output_dataset, input_dataset_test,
                      output_dataset_test)
end_time = time.time()
print('Elapsed time: {} seconds'.format(end_time-start_time))

# test