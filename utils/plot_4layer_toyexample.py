import numpy as np
import matplotlib.pyplot as plt

log_dir = '../logs/toy_example_normal_DTP_4layers/'
max_epoch = 600

# ===== Save results =======
train_losses_true = np.load(log_dir + 'train_losses_true.npy')
test_losses_true = np.load(log_dir + 'test_losses_true.npy')
train_losses_false = np.load(log_dir + 'train_losses_false.npy')
test_losses_false = np.load(log_dir + 'test_losses_false.npy')
train_losses_BP = np.load(log_dir + 'train_losses_BP.npy')
test_losses_BP = np.load(log_dir + 'test_losses_BP.npy')
train_losses_BPfixed = np.load(log_dir + 'train_losses_BPfixed.npy')
test_losses_BPfixed = np.load(log_dir + 'test_losses_BPfixed.npy')

# ====== Average results =======
train_loss_true_mean = np.mean(train_losses_true, axis=0)
train_loss_false_mean = np.mean(train_losses_false, axis=0)
test_loss_true_mean = np.mean(test_losses_true, axis=0)
test_loss_false_mean = np.mean(test_losses_false, axis=0)
train_loss_BP_mean = np.mean(train_losses_BP, axis=0)
train_loss_BPfixed_mean = np.mean(train_losses_BPfixed, axis=0)
test_loss_BP_mean = np.mean(test_losses_BP, axis=0)
test_loss_BPfixed_mean = np.mean(test_losses_BPfixed, axis=0)

# ==== Plots ======
fontsize = 14
epochs = np.arange(0, max_epoch+1)
legend1 = ['RTP-AI', 'TP-AI', 'BP', 'fixed BP']
legend2 = ['TP', 'DTP', 'original TP', 'original DTP']
# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure('training_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, train_loss_true_mean)
plt.semilogy(epochs, train_loss_false_mean)
plt.semilogy(epochs, train_loss_BP_mean)
plt.semilogy(epochs, train_loss_BPfixed_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1, fontsize=fontsize)
plt.show()

fig = plt.figure('test_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, test_loss_true_mean)
plt.semilogy(epochs, test_loss_false_mean)
plt.semilogy(epochs, test_loss_BP_mean)
plt.semilogy(epochs, test_loss_BPfixed_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1, fontsize=fontsize)
plt.show()