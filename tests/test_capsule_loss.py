from layers.layer import CapsuleOutputLayer
import torch
import numpy as np
from tensorboardX import SummaryWriter
import utils.helper_functions as hf
import random
import matplotlib.pyplot as plt

seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# User variables
n = 20
nb_classes = 5
batch_size = 10
h = 1e-3
finite_differences = True
custom_implementation = False
custom_implementation2 = False

if finite_differences:
    m_min = 0.1
    m_plus = 0.9
    l = 0.5
    writer = SummaryWriter()

    layer = CapsuleOutputLayer(n,n,nb_classes,writer=writer)

    targets = torch.randint(0,nb_classes,(batch_size,))
    targets = hf.one_hot(targets, nb_classes)

    activation = torch.randn(batch_size,n,1)
    layer.forward_linear_activation = activation
    layer.compute_capsules()

    layer.compute_backward_output(targets)
    gradient = layer.backward_output

    loss = layer.loss(targets)
    gradient_fd = torch.empty(activation.shape)

    # compute finite difference gradient
    for b in range(batch_size):
        for i in range(n):
            fd = torch.zeros(activation.shape)
            fd[b,i,0] = h
            activation_fd = activation + fd
            layer.forward_linear_activation = activation_fd
            layer.compute_capsules()
            loss_fd = layer.loss(targets)
            gradient_fd[b,i,0] = (loss_fd-loss)/h

    error = torch.norm(gradient-gradient_fd, p=float('inf'))

    print('error: {}'.format(error))
    print(gradient)
    print(gradient_fd)
    print(gradient/gradient_fd)

if custom_implementation:
    n = 5
    s = 10*torch.rand(n)
    s_base = s
    v = s**2/(1+s**2)
    m_min = 0.1
    m_plus = 0.9
    l = 0.5
    targets = torch.zeros((n,))
    targets[0] = 1
    h = 1e-3

    Lk = targets*torch.max(torch.stack([m_plus-v,
                torch.zeros(v.shape)]), dim=0)[0]**2 + \
                l*(1-targets)*torch.max(torch.stack([v-m_min,
                torch.zeros(v.shape)]), dim=0)[0]**2
    L = torch.sum(Lk)

    Lk_vk = -2*targets*torch.max(torch.stack([m_plus-v,
                torch.zeros(v.shape)]), dim=0)[0]+ \
                2*l*(1-targets)*torch.max(torch.stack([v \
                                                      -m_min,
                torch.zeros(v.shape)]), dim=0)[0]
    vk_sk = 1/(1+s**2)**2 * 2* s
    gradient = Lk_vk*vk_sk

    gradient_fd = torch.empty(gradient.shape)
    for i in range(n):
        fd = torch.zeros((n,))
        fd[i] = h
        s = s_base + fd
        v = s ** 2 / (1 + s ** 2)
        Lk_fd = targets * torch.max(torch.stack([m_plus - v,
                                              torch.zeros(v.shape)]), dim=0)[
            0] ** 2 + \
             l * (1 - targets) * torch.max(torch.stack([v - m_min,
                                                        torch.zeros(v.shape)]),
                                           dim=0)[0] ** 2
        L_fd = torch.sum(Lk_fd)
        gradient_fd[i] = (L_fd-L)/h

    error = torch.norm(gradient-gradient_fd, p=float('inf'))
    print('error: {}'.format(error))
    print(gradient)
    print(gradient_fd)

if custom_implementation2:
    errors = np.array([])
    iter = 100
    relative_errors = np.array([])
    h = 3e-1
    for i in range(iter):
        n = 5
        size = 100
        batch_size = 1
        s = 10*torch.rand(batch_size, n, size)
        s_base = s
        s_magnitude = torch.norm(s, dim=2)
        s_magnitude = s_magnitude.reshape((s_magnitude.shape[0],
                                           s_magnitude.shape[1], 1))
        v = s_magnitude**2/(1+s_magnitude**2)
        m_min = 0.1
        m_plus = 0.9
        l = 0.5
        targets = torch.randint(0, n, (batch_size,))
        targets = hf.one_hot(targets, n)


        Lk = targets*torch.max(torch.stack([m_plus-v,
                    torch.zeros(v.shape)]), dim=0)[0]**2 + \
                    l*(1-targets)*torch.max(torch.stack([v-m_min,
                    torch.zeros(v.shape)]), dim=0)[0]**2
        L = torch.sum(Lk)

        Lk_vk = -2*targets*torch.max(torch.stack([m_plus-v,
                    torch.zeros(v.shape)]), dim=0)[0]+ \
                    2*l*(1-targets)*torch.max(torch.stack([v \
                                                          -m_min,
                    torch.zeros(v.shape)]), dim=0)[0]
        vk_sk = 1/(1+s_magnitude**2)**2 * 2* s
        gradient = Lk_vk*vk_sk

        gradient_fd = torch.empty(gradient.shape)
        for b in range(batch_size):
            for i in range(n):
                for j in range(size):
                    fd = torch.zeros((s.shape))
                    fd[b,i,j] = h
                    s = s_base + fd
                    s_magnitude = torch.norm(s, dim=2)
                    s_magnitude = s_magnitude.reshape((s_magnitude.shape[0],
                                               s_magnitude.shape[1], 1))
                    v = s_magnitude ** 2 / (1 + s_magnitude ** 2)
                    Lk_fd = targets * torch.max(torch.stack([m_plus - v,
                                                          torch.zeros(v.shape)]), dim=0)[
                        0] ** 2 + \
                         l * (1 - targets) * torch.max(torch.stack([v - m_min,
                                                                    torch.zeros(v.shape)]),
                                                       dim=0)[0] ** 2
                    L_fd = torch.sum(Lk_fd)
                    gradient_fd[b, i,j] = (L_fd-L)/h

        error = torch.norm(gradient-gradient_fd, p=float('inf'))
        relative_error = error/torch.norm(gradient, p=float('inf'))
        errors = np.append(errors,error)
        relative_errors = np.append(relative_errors,relative_error)


    print('max error: {}'.format(np.max(errors)))
    print('max relative error: {}'.format(np.max(relative_errors)))
    plt.figure()
    plt.hist(errors)
    plt.title('errors')
    plt.show()

    plt.figure()
    plt.hist(relative_errors)
    plt.title('relative errors')
    plt.show()
    # print(gradient)
    # print(gradient_fd)

