import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import utils.helper_functions as hf
from layers.invertible_layer import InvertibleLayer, InvertibleOutputLayer
from tensorboardX import SummaryWriter
from utils.helper_classes import TestError

seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# User variables
n = 5
learning_rate = 0.01
epsilon = 0.1

# ======== set log directory ==========
log_dir = '../logs/debug_TP'
writer = SummaryWriter(log_dir=log_dir)

hidden_layer = InvertibleLayer(n,n,n,writer,name='hidden_layer',
                               epsilon=epsilon)
output_layer = InvertibleOutputLayer(n, n, writer, step_size=0.01)


# tests
s_mins = np.array([])
ds = np.array([])
betas = np.array([])
errors = np.array([])
forward_errors = np.array([])

for i in range(20000):
    A = 5 * hf.get_invertible_random_matrix(n, n)
    u = -0.5/learning_rate * torch.randn(n, 1)
    v = 0.5 * torch.randn(n, 1)
    A_inv = torch.inverse(A)
    weight_gradients = torch.matmul(u, torch.transpose(v, -1, -2))

    output_layer.set_forward_parameters(A, torch.zeros(n,1))
    hidden_layer.set_backward_parameters(A_inv, torch.zeros(n,1))
    output_layer.set_weight_update_u(u)
    output_layer.set_weight_update_v(v)
    output_layer.set_forward_gradients(weight_gradients, torch.zeros(n,1))
    output_layer.update_forward_parameters(learning_rate)
    hidden_layer.update_backward_parameters(learning_rate, output_layer)
    At = output_layer.forward_weights
    At_inv_SM = hidden_layer.backward_weights
    error = torch.norm(torch.eye(n) - At_inv_SM * At)
    errors = np.append(errors, error)
    U, S, V = torch.svd(At)
    s_min = S[-1]
    s_mins = np.append(s_mins, s_min)
    ds = np.append(ds, torch.abs(1+hidden_layer.d))
    betas = np.append(betas, hidden_layer.beta)
    At_own = A + hidden_layer.beta * torch.matmul(-learning_rate*u,
                                              torch.transpose(v, -1, -2))
    forward_error = torch.norm(At-At_own)
    forward_errors = np.append(forward_errors, forward_error)

errors = errors/(n)
forward_errors = forward_errors/(n)


plt.figure()
plt.loglog(ds, s_mins, '*')
plt.title(r'$s_{min}$ vs $d$ (pyprop)')
plt.xlabel(r'$|d+1|$')
plt.ylabel(r'$s_{min}$')
plt.show()

plt.figure()
plt.hist(betas, bins=30)
plt.title(r'$\beta$ (pyprop)')
plt.show()

plt.figure()
plt.hist(errors, bins=30)
plt.title('inverse errors (pyplot)')
plt.show()

plt.figure()
plt.hist(errors*s_mins, bins=50)
plt.title('inverse error times $s_{min}$')
plt.show()

plt.figure()
plt.loglog(ds, errors, '*')
plt.title('errors vs $d$ (pyplot)')
plt.show()

plt.figure()
plt.figure()
plt.hist(forward_errors, bins=30)
plt.title('forward errors (pyplot)')
plt.show()

# Numerical tests:
if np.max(forward_errors) < 1e-5:
    print('Forward weights update with beta OK')
else:
    raise(TestError('Forward weight updates with beta failed'))







