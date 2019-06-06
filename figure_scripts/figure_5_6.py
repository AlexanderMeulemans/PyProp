import torch
from utils import helper_functions as hf
import matplotlib.pyplot as plt

# Parameter settings
iterations = 10000
nb_blocks = 2
block_size = 10
batch = 1
s = 1.

"""
figure 5.5a:
iterations = 10000
nb_blocks = 2
block_size = 10
batch = 1
s = 1.

figure 5.5b:
iterations = 10000
nb_blocks = 10
block_size = 10
batch = 1
s = 1.

figure 5.5b:
iterations = 10000
nb_blocks = 10
block_size = 10
batch = 100
s = 1.

"""

# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


angles = torch.empty(iterations)
angles_random = torch.empty(iterations)
angles_bp_block = torch.empty(iterations)
angles_bp_gn = torch.empty(iterations)
angles_random_v = torch.empty(iterations)
angles_block_gn_rand = torch.empty(iterations)

for iter in range(iterations):
    if iter%100==0:
        print(iter)
    error = torch.randn(block_size * batch, 1)
    layer_jacobians = torch.empty(batch,nb_blocks,block_size,block_size)
    layer_jacobians_random = s*torch.randn(layer_jacobians.shape)
    for b in range(batch):
        J = torch.eye(block_size)
        for i in range(nb_blocks):
            J_new = s*torch.randn(block_size,block_size)
            J = torch.matmul(J,J_new)
            layer_jacobians[b,nb_blocks-i-1,:,:] = J

    # Construct GN hessian
    J_tot = torch.empty(block_size*batch, nb_blocks*block_size)
    J_tot_rand = torch.empty(J_tot.shape)
    for b in range(batch):
        for i in range(nb_blocks):
            J_tot[b*block_size:(b+1)*block_size,i*block_size:(i+1)*block_size] = layer_jacobians[b,i,:,:]
            J_tot_rand[b * block_size:(b + 1) * block_size,
            i * block_size:(i + 1) * block_size] = layer_jacobians_random[b, i, :, :]

    J_tot_pinv = torch.pinverse(J_tot)
    J_tot_rand_pinv = torch.pinverse(J_tot_rand)
    h_update = torch.matmul(J_tot_pinv, error)
    h_update_random = torch.matmul(J_tot_rand_pinv, error)

    h_update_blockdiag = torch.empty(h_update.shape)
    h_update_blockdiag_rand = torch.empty(h_update.shape)
    for i in range(nb_blocks):
        J_i_pinv = torch.pinverse(J_tot[:,i*block_size:(i+1)*block_size])
        J_i_rand_pinv = torch.pinverse(J_tot_rand[:,i*block_size:(i+1)*block_size])
        h_update_blockdiag[i*block_size:(i+1)*block_size] = torch.matmul(J_i_pinv, error)
        h_update_blockdiag_rand[i*block_size:(i+1)*block_size] = torch.matmul(J_i_rand_pinv, error)

    h_update = h_update.unsqueeze(0)
    h_update_random = h_update_random.unsqueeze(0)
    h_update_blockdiag = h_update_blockdiag.unsqueeze(0)
    h_update_blockdiag_rand = h_update_blockdiag_rand.unsqueeze(0)
    angle = hf.get_angle(h_update,h_update_blockdiag)
    angle_block_gn_rand = hf.get_angle(h_update_random, h_update_blockdiag_rand)
    angles[iter] = angle
    angles_block_gn_rand[iter] = angle_block_gn_rand

    h_backprop = torch.matmul(torch.transpose(J_tot,0,1), error)
    h_backprop = h_backprop.unsqueeze(0)

    angle_bp_block = hf.get_angle(h_update_blockdiag, h_backprop)
    angles_bp_block[iter] = angle_bp_block
    angle_bp_gn = hf.get_angle(h_update, h_backprop)
    angles_bp_gn[iter] = angle_bp_gn

    # Random control
    J_random1 = s*torch.randn(J_tot_pinv.shape)
    J_random2 = s*torch.randn(J_tot_pinv.shape)

    h_random1 = torch.matmul(J_random1, error)
    h_random2 = torch.matmul(J_random2, error)
    h_random1 = h_random1.unsqueeze(0)
    h_random2 = h_random2.unsqueeze(0)
    angle_random = hf.get_angle(h_random1,h_random2)
    angles_random[iter] = angle_random

    h_random_v1 = torch.randn(h_random1.shape)
    h_random_v2 = torch.randn(h_random1.shape)
    angle_random_v = hf.get_angle(h_random1, h_random2)
    angles_random_v[iter] = angle_random_v

fig = plt.figure('angle block approx gn')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=21)
plt.hist(angles,bins=100)
plt.xlabel(r'$\cos(\alpha)$', fontsize=26)
plt.ylabel(r'\# samples', fontsize=26)

plt.show()
# plt.figure('angle BP block approx')
# plt.hist(angles_bp_block, bins=100)
# plt.title('angle BP block approx')
# plt.show()
# plt.figure()
# plt.hist(angles_bp_gn, bins=100)
# plt.title('angle BP GN')
# plt.show()
# plt.figure('angle random control')
# plt.hist(angles_random, bins=100)
# plt.title('angle random control')
# plt.show()
# plt.figure()
# plt.hist(angles_block_gn_rand, bins=100)
# plt.title('angle block gn random')
# plt.show()
# plt.figure()
# plt.hist(angles_random_v, bins=100)
# plt.title('random vectors')
# plt.show()









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



