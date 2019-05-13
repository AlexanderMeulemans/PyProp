import torch
from layers.invertible_layer import InvertibleInputLayer, InvertibleLeakyReluLayer, InvertibleLinearOutputLayer
from optimizers.optimizers import SGDInvertible
from networks.invertible_network import InvertibleNetwork
from utils.create_datasets import GenerateDatasetFromModel
from tensorboardX import SummaryWriter
from utils import helper_functions as hf


n=3
writer = SummaryWriter()
epsilon = 1e-5



input_layer = InvertibleInputLayer(layer_dim=n, out_dim=n, loss_function='mse',
                                  name='input_layer', writer=writer)
hidden_layer = InvertibleLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                       layer_dim=n, out_dim=n, loss_function=
                                       'mse',
                                       name='hidden_layer',
                                       writer=writer)
hidden_layer2 = InvertibleLeakyReluLayer(negative_slope=0.35, in_dim=n,
                                       layer_dim=n, out_dim=n, loss_function=
                                       'mse',
                                       name='hidden_layer',
                                       writer=writer)
output_layer = InvertibleLinearOutputLayer(in_dim=n, layer_dim=n,
                                          step_size=0.01,
                                          name='output_layer',
                                          writer=writer)
network = InvertibleNetwork([input_layer, hidden_layer, hidden_layer2,
                             output_layer])

# Initializing optimizer
optimizer3 = SGDInvertible(network=network, threshold=0.000001,
                           init_step_size=0.02, tau=50,
                           final_step_size=0.018,
                           learning_rate=0.5, max_epoch=1)

generator = GenerateDatasetFromModel(network)

input_dataset, output_dataset = generator.generate(1,
                                                   1)
input_dataset_test, output_dataset_test = generator.generate(
    1, 1)

optimizer3.run_dataset(input_dataset, output_dataset, input_dataset_test,
                       output_dataset_test)
J = network.compute_total_jacobian()
h1 = hidden_layer.forward_output
h2 = hidden_layer2.forward_output
output = output_layer.forward_output

J_fd = torch.empty(J.shape)
for j in range(J.shape[1]):
    if j < h1.shape[1]:
        fd1 = torch.zeros(h1.shape)
        fd1[0,j,0] = epsilon
        h1_fd = h1 + fd1
        hidden_layer.set_forward_output(h1_fd)
        hidden_layer2.propagate_forward(hidden_layer)
        output_layer.propagate_forward(hidden_layer2)
        output_fd = output_layer.forward_output
        output_difference = output_fd - output
        J_fd[:,j] = output_difference.squeeze()/epsilon
    else:
        fd2 = torch.zeros(h2.shape)
        fd2[0,j-h1.shape[1],0] = epsilon
        h2_fd = h2 + fd2
        hidden_layer2.set_forward_output(h2_fd)
        output_layer.propagate_forward(hidden_layer2)
        output_fd = output_layer.forward_output
        output_difference = output_fd - output
        J_fd[:, j] = output_difference.squeeze() / epsilon

error = J-J_fd
max_error = torch.max(error)
print('max error: {}'.format(max_error))
