import numpy as np
import matplotlib.pyplot as plt
import random

seed = 47
np.random.seed(seed)
random.seed(seed)

# User variables
eta = 0.5
eta_tp =1.0
training_iter = 500
batch_size = 1
gridpoints = 100
fontsize = 26

# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

w1_true = np.random.randn(2)
w2_fixed = np.random.randn(2,2)


# w1_start = np.array([-1.5,-0.8])
w1_start = np.array([-1.8,-0.7])
w1_train_TP = np.zeros((2,training_iter))
w1_train_BP = np.zeros((2,training_iter))
w1_train_GN = np.zeros((2,training_iter))
w1_train_TP[:,0] = w1_start
w1_train_BP[:,0] = w1_start
w1_train_GN[:,0] = w1_start
xs = np.random.randn(batch_size)


# def lrelu(x):
#     return 1 / (1 + np.exp(-x))

def lrelu(x):
    output = np.zeros(x.shape)
    if x[0]>0:
        output[0] = x[0]
    else:
        output[0] = 0.1*x[0]
    if x[1]>0:
        output[1] = x[1]
    else:
        output[1] = 0.1*x[1]
    return output

def lrelu_derivative(x):
    output = np.zeros(x.shape)
    if x[0] > 0:
        output[0] = 1.
    else:
        output[0] = 0.1
    if x[1] > 0:
        output[1] = 1.
    else:
        output[1] = 0.1
    return output

L = 0
w1 = w1_start
for x in xs:
    l = np.dot(w2_fixed, lrelu(w1 * x) - lrelu(w1_true * x))
    L += 1/float(batch_size)*np.sum(l*l, 0)


def loss(w1, w1_true, w2_fixed, xs):
    L = 0
    for x in xs:
        l = np.dot(w2_fixed, lrelu(w1 * x) - lrelu(w1_true * x))
        L += 1 / float(batch_size) * np.sum(l * l, 0)
    return L

def plot_contours(w1_true, w2_fixed, xs):
    w11 = np.linspace(w1_true[0]-1.5, w1_true[0] + 2.0, gridpoints)
    w12 = np.linspace(w1_true[1]-2.5, w1_true[1] + 4.0, gridpoints)
    X, Y = np.meshgrid(w11, w12)
    Z = np.zeros(X.shape)

    for i in range(gridpoints):
        for j in range(gridpoints):
            w1 = np.array([X[i,j], Y[i,j]])
            Z[i,j] = loss(w1, w1_true,w2_fixed,xs)

    levels = [0.001, 0.01,0.1,0.2, 0.5, 1, 1.5,2.]
    # plt.figure()
    # plt.contour(X,Y,Z)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels)
    # ax.clabel(CS, inline=1, fontsize=10)

def BP_update(w1, w1_true, w2_fixed, xs, eta):
    update_bp = np.zeros(w1.shape)
    for x in xs:
        # sigmoid_der = lrelu(w1*x)*(1-lrelu(w1*x))
        sigmoid_der = lrelu_derivative(w1 * x)
        error = lrelu(w1 * x) - lrelu(w1_true * x)
        update_bp += -1/batch_size*eta*x*sigmoid_der*np.dot(
            np.transpose(w2_fixed),np.dot(w2_fixed, error))
    return update_bp

def TP_update(w1, w1_true, w2_fixed, xs, eta):
    update_tp = np.zeros(w1.shape)
    for x in xs:
        # sigmoid_der = lrelu(w1*x)*(1-lrelu(w1*x))
        sigmoid_der = lrelu_derivative(w1 * x)
        error = lrelu(w1 * x) - lrelu(w1_true * x)
        update_tp += -1 / batch_size *x* eta * sigmoid_der * error
    return update_tp

def GN_update(w1, w1_true, w2_fixed, xs):
    H = np.zeros((2,2))
    gradient = np.zeros(w1.shape)
    for x in xs:
        # sigmoid_der = lrelu(w1*x)*(1-lrelu(w1*x))
        sigmoid_der = lrelu_derivative(w1 * x)
        A = np.dot(w2_fixed, np.diag(sigmoid_der))
        H += 1/batch_size*x**2*np.dot(np.transpose(A),A)
        error = lrelu(w1 * x) - lrelu(w1_true * x)
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
# plt.title('Error back propagation', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = fontsize)
plt.ylabel('$w_{12}$', fontsize=fontsize)
plt.show()

fig2 = plt.figure()
plot_contours(w1_true, w2_fixed, xs)
plt.plot(w1_train_TP[0,:], w1_train_TP[1,:], 'r*-')
# plt.title('Target propagation', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = fontsize)
plt.ylabel('$w_{12}$', fontsize=fontsize)
plt.show()

fig3 = plt.figure()
plot_contours(w1_true, w2_fixed, xs)
plt.plot(w1_train_GN[0,:], w1_train_GN[1,:], 'r*-')
# plt.title('Gauss Newton', fontsize=20)
plt.xlabel('$w_{11}$', fontsize = fontsize)
plt.ylabel('$w_{12}$', fontsize=fontsize)
plt.show()


# def loss_landscape()
