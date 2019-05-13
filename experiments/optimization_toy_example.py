import numpy as np
import matplotlib.pyplot as plt
import random

seed = 47
np.random.seed(seed)
random.seed(seed)

# User variables
eta = 0.3
eta_tp = 0.5
training_iter = 100
batch_size = 64

# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

w1_true = np.random.randn(2)
w2_fixed = np.random.randn(2,2)


w1_start = np.array([-1.8,-1.3])
w1_train_TP = np.zeros((2,training_iter))
w1_train_BP = np.zeros((2,training_iter))
w1_train_TP[:,0] = w1_start
w1_train_BP[:,0] = w1_start




def plot_loss_ellipses(w1_true, w2_fixed):
    U, S, V = np.linalg.svd(w2_fixed)
    c2s = np.linspace(0.1, 4, 5)
    w1_true_tilde = np.dot(np.transpose(V), w1_true)

    plt.figure()
    for c2 in c2s:
        w11_tilde = np.linspace(w1_true_tilde[0] - c2 / S[0],
                                w1_true_tilde[0] + c2 / S[0], 100)
        w12_tilde = 1 / (S[1]) * np.sqrt(
            c2 ** 2 - S[0] ** 2 * (w11_tilde - w1_true_tilde[0]) ** 2) + \
                    w1_true_tilde[1]
        w12_tilde_min = -1 / (S[1]) * np.sqrt(
            c2 ** 2 - S[0] ** 2 * (w11_tilde - w1_true_tilde[0]) ** 2) + \
                        w1_true_tilde[1]
        w1_tilde_plus = np.stack([w11_tilde, w12_tilde])
        w1_tilde_min = np.stack([w11_tilde, w12_tilde_min])
        w1_plus = np.dot(V, w1_tilde_plus)
        w1_min = np.dot(V, w1_tilde_min)

        plt.plot(w1_plus[0], w1_plus[1], 'b')
        plt.plot(w1_min[0], w1_min[1], 'b')


def loss(w1, w1_true, w2_fixed):
    if len(w1.shape)>1:
        l = np.dot(w2_fixed, w1 - np.expand_dims(w1_true,1))
    else:
        l = np.dot(w2_fixed, w1-w1_true)
    return np.sum(l*l, 0)


for iter in range(1, training_iter):
    x = np.random.randn(batch_size)
    Ex2 = np.mean(x**2)
    update_BP = -eta * Ex2 * np.dot(np.transpose(w2_fixed), np.dot(w2_fixed, w1_train_BP[:,iter-1]-w1_true))
    w1_train_BP[:,iter] = w1_train_BP[:,iter-1] + update_BP

    update_TP = -eta_tp*Ex2*(w1_train_TP[:, iter-1]-w1_true)
    w1_train_TP[:, iter] = w1_train_TP[:,iter-1] + update_TP

w1_train_GN = np.stack([w1_start, w1_true],1)

fig1 = plt.figure()
plot_loss_ellipses(w1_true, w2_fixed)
plt.plot(w1_train_BP[0,:], w1_train_BP[1,:], 'r*-')
plt.title('Error back propagation', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = 20)
plt.ylabel('$w_{12}$', fontsize=20)
plt.show()

fig2 = plt.figure()
plot_loss_ellipses(w1_true, w2_fixed)
plt.plot(w1_train_TP[0,:], w1_train_TP[1,:], 'r*-')
plt.title('Target propagation', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = 20)
plt.ylabel('$w_{12}$', fontsize=20)
plt.show()

fig3 = plt.figure()
plot_loss_ellipses(w1_true, w2_fixed)
plt.plot(w1_train_GN[0,:], w1_train_GN[1,:], 'r*-')
plt.title('Gauss Newton', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = 20)
plt.ylabel('$w_{12}$', fontsize=20)
plt.show()

