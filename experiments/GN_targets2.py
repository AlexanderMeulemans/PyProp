import torch
from layers.invertible_layer import InvertibleInputLayer, InvertibleLeakyReluLayer, InvertibleLinearOutputLayer
from optimizers.optimizers import SGDInvertible
from networks.invertible_network import InvertibleNetwork
from utils.create_datasets import GenerateDatasetFromModel
from tensorboardX import SummaryWriter
from utils import helper_functions as hf
import matplotlib.pyplot as plt

iterations = 10000
nb_blocks = 10
block_size = 5


angles = torch.empty(iterations)
angles_random = torch.empty(iterations)
angles_bp_block = torch.empty(iterations)
angles_bp_gn = torch.empty(iterations)


for iter in range(iterations):

    error = torch.randn(block_size, 1)
    layer_jacobians = torch.empty(nb_blocks,block_size,block_size)
    J = torch.eye(block_size)
    for i in range(nb_blocks):
        J_new = torch.randn(block_size,block_size)
        J = torch.matmul(J,J_new)
        layer_jacobians[nb_blocks-i-1,:,:] = J

    # Construct GN hessian
    J_tot = torch.empty(block_size, nb_blocks*block_size)
    for i in range(nb_blocks):
        J_tot[:,i*block_size:(i+1)*block_size] = layer_jacobians[i,:,:]

    J_tot_pinv = torch.pinverse(J_tot)
    h_update = torch.matmul(J_tot_pinv, error)

    h_update_blockdiag = torch.empty(h_update.shape)
    for i in range(nb_blocks):
        J_i_pinv = torch.pinverse(layer_jacobians[i,:,:])
        h_update_blockdiag[i*block_size:(i+1)*block_size] = torch.matmul(J_i_pinv, error)

    h_update = h_update.unsqueeze(0)
    h_update_blockdiag = h_update_blockdiag.unsqueeze(0)
    angle = hf.get_angle(h_update,h_update_blockdiag)
    angles[iter] = angle

    h_backprop = torch.matmul(torch.transpose(J_tot,0,1), error)
    h_backprop = h_backprop.unsqueeze(0)

    angle_bp_block = hf.get_angle(h_update_blockdiag, h_backprop)
    angles_bp_block[iter] = angle_bp_block
    angle_bp_gn = hf.get_angle(h_update, h_backprop)
    angles_bp_gn[iter] = angle_bp_gn

    # Random control
    J_random1 = torch.randn(J_tot_pinv.shape)
    J_random2 = torch.randn(J_tot_pinv.shape)

    h_random1 = torch.matmul(J_random1, error)
    h_random2 = torch.matmul(J_random2, error)
    h_random1 = h_random1.unsqueeze(0)
    h_random2 = h_random2.unsqueeze(0)
    angle_random = hf.get_angle(h_random1,h_random2)
    angles_random[iter] = angle_random


plt.figure('angle block approx gn')
plt.hist(angles,bins=int(iterations/100))
plt.title('angle block approx gn')
plt.show()
plt.figure('angle BP block approx')
plt.hist(angles_bp_block, bins=int(iterations/100))
plt.title('angle BP block approx')
plt.show()
plt.figure()
plt.hist(angles_bp_gn, bins=int(iterations/100))
plt.title('angle BP GN')
plt.show()
plt.figure('angle random control')
plt.hist(angles_random, bins=int(iterations/100))
plt.title('angle random control')
plt.show()








# GN_hessian = torch.empty(nb_blocks*block_size, nb_blocks*block_size)
# GN_hessian_blockdiag = torch.zeros(nb_blocks*block_size, nb_blocks*block_size)
# for i in range(nb_blocks):
#     J_i = layer_jacobians[i, :, :]
#     GN_hessian_blockdiag[i * block_size:(i + 1) * block_size,
#     i * block_size:(i + 1) * block_size] = torch.matmul(torch.transpose(J_i,0,1),J_i)
#     for j in range(nb_blocks):
#         J_j = layer_jacobians[j,:,:]
#         GN_hessian[i*block_size:(i+1)*block_size,
#         j*block_size:(j+1)*block_size] = torch.matmul(torch.transpose(J_i,0,1),J_j)



