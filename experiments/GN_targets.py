import torch
from layers.invertible_layer import InvertibleInputLayer, InvertibleLeakyReluLayer, InvertibleLinearOutputLayer
from optimizers.optimizers import SGDInvertible
from networks.invertible_network import InvertibleNetwork
from utils.create_datasets import GenerateDatasetFromModel
from tensorboardX import SummaryWriter

J1 = torch.randn(3,3)
J2 = torch.randn(3,3)
g = torch.randn(6,1)
Jtot = torch.empty(3,6)
l = 0.0001
Jtot[:,0:3] = J1
Jtot[:,3:6] = J2
H = torch.matmul(Jtot.transpose(0,1),Jtot)
J_pinverse = torch.pinverse(Jtot,rcond=1e-6)

Htot = torch.empty(6,6)
Htot[0:3,0:3] = torch.matmul(J1.transpose(0,1),J1)
Htot[0:3,3:6] = torch.matmul(J1.transpose(0,1),J2)
Htot[3:6,0:3] = torch.matmul(J2.transpose(0,1),J1)
Htot[3:6,3:6] = torch.matmul(J2.transpose(0,1),J2)

Htot_reg = Htot+l*torch.eye(6,6)

Htot_pinverse = torch.pinverse(Htot, rcond=1e-6)

h = torch.gesv(g, Htot_reg)
h_reg = torch.matmul(Htot_pinverse,g)


U,S,V = torch.svd(Htot)
U,S_reg,V = torch.svd(Htot_reg)

n=3
writer = SummaryWriter()

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
htot = network.compute_GN_targets()