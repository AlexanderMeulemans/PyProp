import numpy as np
import matplotlib.pyplot as plt
import random

seed = 47
np.random.seed(seed)
random.seed(seed)

# User variables
eta = 12.
eta_tp =35.
training_iter = 100
batch_size = 64
gridpoints = 50

# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

w1_true = np.random.randn(2)
w2_fixed = np.random.randn(2,2)


w1_start = np.array([-2.1,-1.1])
w1_train_TP = np.zeros((2,training_iter))
w1_train_BP = np.zeros((2,training_iter))
w1_train_GN = np.zeros((2,training_iter))
w1_train_TP[:,0] = w1_start
w1_train_BP[:,0] = w1_start
w1_train_GN[:,0] = w1_start
xs = np.random.randn(batch_size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

L = 0
w1 = w1_start
for x in xs:
    l = np.dot(w2_fixed, sigmoid(w1*x) - sigmoid(w1_true*x))
    L += 1/float(batch_size)*np.sum(l*l, 0)


def loss(w1, w1_true, w2_fixed, xs):
    L = 0
    for x in xs:
        l = np.dot(w2_fixed, sigmoid(w1 * x) - sigmoid(w1_true * x))
        L += 1 / float(batch_size) * np.sum(l * l, 0)
    return L

def plot_contours(w1_true, w2_fixed, xs):
    w11 = np.linspace(w1_true[0]-3.0, w1_true[0] + 3.0, gridpoints)
    w12 = np.linspace(w1_true[1]-3.0, w1_true[1] + 3.0, gridpoints)
    X, Y = np.meshgrid(w11, w12)
    Z = np.zeros(X.shape)

    for i in range(gridpoints):
        for j in range(gridpoints):
            w1 = np.array([X[i,j], Y[i,j]])
            Z[i,j] = loss(w1, w1_true,w2_fixed,xs)

    levels = [0.001, 0.01, 0.05,0.1,0.2, 0.5, 1, 1.5]
    # plt.figure()
    # plt.contour(X,Y,Z)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels)
    ax.clabel(CS, inline=1, fontsize=10)

def BP_update(w1, w1_true, w2_fixed, xs, eta):
    update_bp = np.zeros(w1.shape)
    for x in xs:
        sigmoid_der = sigmoid(w1*x)*(1-sigmoid(w1*x))
        error = sigmoid(w1*x) - sigmoid(w1_true*x)
        update_bp += -1/batch_size*eta*x*sigmoid_der*np.dot(
            np.transpose(w2_fixed),np.dot(w2_fixed, error))
    return update_bp

def TP_update(w1, w1_true, w2_fixed, xs, eta):
    update_tp = np.zeros(w1.shape)
    for x in xs:
        sigmoid_der = sigmoid(w1 * x) * (1 - sigmoid(w1 * x))
        error = sigmoid(w1 * x) - sigmoid(w1_true * x)
        update_tp += -1 / batch_size *x* eta * sigmoid_der * error
    return update_tp

def GN_update(w1, w1_true, w2_fixed, xs):
    H = np.zeros((2,2))
    gradient = np.zeros(w1.shape)
    for x in xs:
        sigmoid_der = sigmoid(w1 * x) * (1 - sigmoid(w1 * x))
        A = np.dot(w2_fixed, np.diag(sigmoid_der))
        H += 1/batch_size*x**2*np.dot(np.transpose(A),A)
        error = sigmoid(w1 * x) - sigmoid(w1_true * x)
        gradient += 1 / batch_size *x* sigmoid_der * np.dot(
            np.transpose(w2_fixed), np.dot(w2_fixed, error))
    return -np.dot(np.linalg.inv(H),gradient)

for iter in range(1, training_iter):
    update_BP = BP_update(w1_train_BP[:, iter-1],w1_true, w2_fixed,xs,eta)
    w1_train_BP[:,iter] = w1_train_BP[:, iter-1] + update_BP
    update_TP = TP_update(w1_train_TP[:, iter-1], w1_true, w2_fixed, xs, eta_tp)
    w1_train_TP[:, iter] = w1_train_TP[:,iter-1] + update_TP
    update_GN = GN_update(w1_train_GN[:, iter-1],w1_true,w2_fixed,xs)
    w1_train_GN[:, iter] = w1_train_GN[:, iter-1] + update_GN


fig1 = plt.figure()
plot_contours(w1_true, w2_fixed, xs)
plt.plot(w1_train_BP[0,:], w1_train_BP[1,:], 'r*-')
plt.title('Error back propagation', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = 20)
plt.ylabel('$w_{12}$', fontsize=20)
plt.show()

fig2 = plt.figure()
plot_contours(w1_true, w2_fixed, xs)
plt.plot(w1_train_TP[0,:], w1_train_TP[1,:], 'r*-')
plt.title('Target propagation', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = 20)
plt.ylabel('$w_{12}$', fontsize=20)
plt.show()

fig3 = plt.figure()
plot_contours(w1_true, w2_fixed, xs)
plt.plot(w1_train_GN[0,:], w1_train_GN[1,:], 'r*-')
plt.title('Gauss Newton', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = 20)
plt.ylabel('$w_{12}$', fontsize=20)
plt.show()


# def loss_landscape()
