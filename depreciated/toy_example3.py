import torch

from layers.invertible_layer import InvertibleInputLayer, \
    InvertibleLeakyReluLayer, InvertibleSoftmaxOutputLayer, InvertibleNetwork
from optimizers.optimizers import SGD, SGDMomentum
from utils.create_datasets import GenerateDatasetFromModel

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('using GPU')
else:
    print('using CPU')

# Create toy model dataset

input_layer_true = InvertibleInputLayer(layer_dim=6, out_dim=5,
                                        loss_function='mse')
hidden_layer_true = InvertibleLeakyReluLayer(negative_slope=0.01, in_dim=6,
                                             layer_dim=5, out_dim=4,
                                             loss_function=
                                             'mse')
output_layer_true = InvertibleSoftmaxOutputLayer(in_dim=5, layer_dim=4,
                                                 step_size=0.05)

true_network = InvertibleNetwork([input_layer_true, hidden_layer_true,
                                  output_layer_true])

generator = GenerateDatasetFromModel(true_network)

input_dataset, output_dataset = generator.generate(7000, 1)
input_dataset_test, output_dataset_test = generator.generate(1000, 1)

# Creating training network
inputlayer = InvertibleInputLayer(layer_dim=6, out_dim=5, loss_function='mse')
hiddenlayer = InvertibleLeakyReluLayer(negative_slope=0.01, in_dim=6,
                                       layer_dim=5, out_dim=4, loss_function=
                                       'mse')
outputlayer = InvertibleSoftmaxOutputLayer(in_dim=5, layer_dim=4,
                                           step_size=0.05)

network = InvertibleNetwork([inputlayer, hiddenlayer, outputlayer])

# Initializing optimizer
optimizer1 = SGD(network=network, threshold=0.01, init_learning_rate=0.01,
                 tau=100,
                 final_learning_rate=0.0005, compute_accuracies=False,
                 max_epoch=120)
optimizer2 = SGDMomentum(network=network, threshold=1.2, init_learning_rate=0.1,
                         tau=100, final_learning_rate=0.005,
                         compute_accuracies=False, max_epoch=150, momentum=0.5)

# Train on MNIST
optimizer1.run_dataset(input_dataset, output_dataset)

# Test network

predicted_classes = network.predict(input_dataset_test[0, :, :, :])
test_loss = network.loss(output_dataset_test[0, :, :, :])
print('Test Loss: ' + str(test_loss))
