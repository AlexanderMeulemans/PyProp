import numpy as np
import matplotlib.pyplot as plt

# log_dir_main = '../figure_data/final_combined_toy_example/'
log_dir_main = '../logs/final_combined_toy_example_unequal_layers_save/'
# log_dir_save = '../logs/final_combined_toy_example_unequal_layers_save/'
max_epoch = 60

indices = [0,1,2,4,5,6,7,8,9,10,11,13]

# ====== load results =======
# ====== Save results =======
train_losses_TP = np.load(log_dir_main + 'train_losses_TP.npy')
test_losses_TP=np.load(log_dir_main + 'test_losses_TP.npy')
approx_error_angles_array_TP=np.load(log_dir_main + 'approx_error_angles_array_TP.npy')
approx_errors_array_TP=np.load(log_dir_main + 'approx_errors_array_TP.npy')
GN_errors_array_TP=np.load(log_dir_main + 'GN_errors_array_TP.npy')
TP_errors_array_TP=np.load(log_dir_main + 'TP_errors_array_TP.npy')
GN_angles_array_TP=np.load(log_dir_main + 'GN_angles_array_TP.npy')
BP_angles_array_TP=np.load(log_dir_main + 'BP_angles_array_TP.npy')

train_losses_DTP=np.load(log_dir_main + 'train_losses_DTP.npy')
test_losses_DTP=np.load(log_dir_main + 'test_losses_DTP.npy')
approx_error_angles_array_DTP=np.load(log_dir_main + 'approx_error_angles_array_DTP.npy')
approx_errors_array_DTP=np.load(log_dir_main + 'approx_errors_array_DTP.npy')
GN_errors_array_DTP=np.load(log_dir_main + 'GN_errors_array_DTP.npy')
TP_errors_array_DTP=np.load(log_dir_main + 'TP_errors_array_DTP.npy')
GN_angles_array_DTP=np.load(log_dir_main + 'GN_angles_array_DTP.npy')
BP_angles_array_DTP=np.load(log_dir_main + 'BP_angles_array_DTP.npy')

train_losses_originalTP=np.load(log_dir_main + 'train_losses_originalTP.npy')
test_losses_originalTP=np.load(log_dir_main + 'test_losses_originalTP.npy')
approx_error_angles_array_originalTP=np.load(log_dir_main + 'approx_error_angles_array_originalTP.npy')
approx_errors_array_originalTP=np.load(log_dir_main + 'approx_errors_array_originalTP.npy')
GN_errors_array_originalTP=np.load(log_dir_main + 'GN_errors_array_originalTP.npy')
TP_errors_array_originalTP=np.load(log_dir_main + 'TP_errors_array_originalTP.npy')
GN_angles_array_originalTP=np.load(log_dir_main + 'GN_angles_array_originalTP.npy')
BP_angles_array_originalTP=np.load(log_dir_main + 'BP_angles_array_originalTP.npy')

train_losses_originalDTP=np.load(log_dir_main + 'train_losses_originalDTP.npy')
test_losses_originalDTP=np.load(log_dir_main + 'test_losses_originalDTP.npy')
approx_error_angles_array_originalDTP=np.load(log_dir_main + 'approx_error_angles_array_originalDTP.npy')
approx_errors_array_originalDTP=np.load(log_dir_main + 'approx_errors_array_originalDTP.npy')
GN_errors_array_originalDTP=np.load(log_dir_main + 'GN_errors_array_originalDTP.npy')
TP_errors_array_originalDTP=np.load(log_dir_main + 'TP_errors_array_originalDTP.npy')
GN_angles_array_originalDTP=np.load(log_dir_main + 'GN_angles_array_originalDTP.npy')
BP_angles_array_originalDTP=np.load(log_dir_main + 'BP_angles_array_originalDTP.npy')

train_losses_BP=np.load(log_dir_main + 'train_losses_BP.npy')
test_losses_BP=np.load(log_dir_main + 'test_losses_BP.npy')

train_losses_BP_fixed=np.load(log_dir_main + 'train_losses_BP_fixed.npy')
test_losses_BP_fixed=np.load(log_dir_main + 'test_losses_BP_fixed.npy')

# train_losses_TP = train_losses_TP[indices,:]
# test_losses_TP=test_losses_TP[indices,:]
# approx_error_angles_array_TP=approx_error_angles_array_TP[indices,:]
# approx_errors_array_TP=approx_errors_array_TP[indices,:]
# GN_errors_array_TP=GN_errors_array_TP[indices,:]
# TP_errors_array_TP=TP_errors_array_TP[indices,:]
# GN_angles_array_TP=GN_angles_array_TP[indices,:]
# BP_angles_array_TP=BP_angles_array_TP[indices,:]
#
# train_losses_DTP=train_losses_DTP[indices,:]
# test_losses_DTP=test_losses_DTP[indices,:]
# approx_error_angles_array_DTP=approx_error_angles_array_DTP[indices,:]
# approx_errors_array_DTP=approx_errors_array_DTP[indices,:]
# GN_errors_array_DTP=GN_errors_array_DTP[indices,:]
# TP_errors_array_DTP=TP_errors_array_DTP[indices,:]
# GN_angles_array_DTP=GN_angles_array_DTP[indices,:]
# BP_angles_array_DTP=BP_angles_array_DTP[indices,:]
#
# train_losses_originalTP=train_losses_originalTP[indices,:]
# test_losses_originalTP=test_losses_originalTP[indices,:]
# approx_error_angles_array_originalTP=approx_error_angles_array_originalTP[indices,:]
# approx_errors_array_originalTP=approx_errors_array_originalTP[indices,:]
# GN_errors_array_originalTP=GN_errors_array_originalTP[indices,:]
# TP_errors_array_originalTP=TP_errors_array_originalTP[indices,:]
# GN_angles_array_originalTP=GN_angles_array_originalTP[indices,:]
# BP_angles_array_originalTP=BP_angles_array_originalTP[indices,:]
#
# train_losses_originalDTP=train_losses_originalDTP[indices,:]
# test_losses_originalDTP=test_losses_originalDTP[indices,:]
# approx_error_angles_array_originalDTP=approx_error_angles_array_originalDTP[indices,:]
# approx_errors_array_originalDTP=approx_errors_array_originalDTP[indices,:]
# GN_errors_array_originalDTP=GN_errors_array_originalDTP[indices,:]
# TP_errors_array_originalDTP=TP_errors_array_originalDTP[indices,:]
# GN_angles_array_originalDTP=GN_angles_array_originalDTP[indices,:]
# BP_angles_array_originalDTP=BP_angles_array_originalDTP[indices,:]
#
#
# train_losses_BP=train_losses_BP[indices,:]
# test_losses_BP=test_losses_BP[indices,:]
#
# train_losses_BP_fixed=train_losses_BP_fixed[indices,:]
# test_losses_BP_fixed=test_losses_BP_fixed[indices,:]


# # ====== Save results =======
# np.save(log_dir_save + 'train_losses_TP.npy', train_losses_TP)
# np.save(log_dir_save + 'test_losses_TP.npy', test_losses_TP)
# np.save(log_dir_save + 'approx_error_angles_array_TP.npy', approx_error_angles_array_TP)
# np.save(log_dir_save + 'approx_errors_array_TP.npy', approx_errors_array_TP)
# np.save(log_dir_save + 'GN_errors_array_TP.npy', GN_errors_array_TP)
# np.save(log_dir_save + 'TP_errors_array_TP.npy', TP_errors_array_TP)
# np.save(log_dir_save + 'GN_angles_array_TP.npy', GN_angles_array_TP)
# np.save(log_dir_save + 'BP_angles_array_TP.npy', BP_angles_array_TP)
#
# np.save(log_dir_save + 'train_losses_DTP.npy', train_losses_DTP)
# np.save(log_dir_save + 'test_losses_DTP.npy', test_losses_DTP)
# np.save(log_dir_save + 'approx_error_angles_array_DTP.npy', approx_error_angles_array_DTP)
# np.save(log_dir_save + 'approx_errors_array_DTP.npy', approx_errors_array_DTP)
# np.save(log_dir_save + 'GN_errors_array_DTP.npy', GN_errors_array_DTP)
# np.save(log_dir_save + 'TP_errors_array_DTP.npy', TP_errors_array_DTP)
# np.save(log_dir_save + 'GN_angles_array_DTP.npy', GN_angles_array_DTP)
# np.save(log_dir_save + 'BP_angles_array_DTP.npy', BP_angles_array_DTP)
#
# np.save(log_dir_save + 'train_losses_originalTP.npy', train_losses_originalTP)
# np.save(log_dir_save + 'test_losses_originalTP.npy', test_losses_originalTP)
# np.save(log_dir_save + 'approx_error_angles_array_originalTP.npy', approx_error_angles_array_originalTP)
# np.save(log_dir_save + 'approx_errors_array_originalTP.npy', approx_errors_array_originalTP)
# np.save(log_dir_save + 'GN_errors_array_originalTP.npy', GN_errors_array_originalTP)
# np.save(log_dir_save + 'TP_errors_array_originalTP.npy', TP_errors_array_originalTP)
# np.save(log_dir_save + 'GN_angles_array_originalTP.npy', GN_angles_array_originalTP)
# np.save(log_dir_save + 'BP_angles_array_originalTP.npy', BP_angles_array_originalTP)
#
# np.save(log_dir_save + 'train_losses_originalDTP.npy', train_losses_originalDTP)
# np.save(log_dir_save + 'test_losses_originalDTP.npy', test_losses_originalDTP)
# np.save(log_dir_save + 'approx_error_angles_array_originalDTP.npy', approx_error_angles_array_originalDTP)
# np.save(log_dir_save + 'approx_errors_array_originalDTP.npy', approx_errors_array_originalDTP)
# np.save(log_dir_save + 'GN_errors_array_originalDTP.npy', GN_errors_array_originalDTP)
# np.save(log_dir_save + 'TP_errors_array_originalDTP.npy', TP_errors_array_originalDTP)
# np.save(log_dir_save + 'GN_angles_array_originalDTP.npy', GN_angles_array_originalDTP)
# np.save(log_dir_save + 'BP_angles_array_originalDTP.npy', BP_angles_array_originalDTP)
#
# np.save(log_dir_save + 'train_losses_BP.npy', train_losses_BP)
# np.save(log_dir_save + 'test_losses_BP.npy', test_losses_BP)
#
# np.save(log_dir_save + 'train_losses_BP_fixed.npy', train_losses_BP_fixed)
# np.save(log_dir_save + 'test_losses_BP_fixed.npy', test_losses_BP_fixed)

# ========= Average results ==========
train_loss_TP_mean = np.mean(train_losses_TP, axis=0)
test_loss_TP_mean = np.mean(test_losses_TP, axis=0)
approx_errors_TP_mean = np.mean(approx_errors_array_TP, axis=0)
approx_error_angle_TP_mean = np.mean(approx_error_angles_array_TP, axis=0)
inverse_fraction_learning_signal_TP = np.mean(approx_errors_array_TP/GN_errors_array_TP, axis=0)
GN_errors_TP_mean = np.mean(GN_errors_array_TP, axis=0)
GN_angles_TP_mean = np.mean(GN_angles_array_TP, axis=0)
BP_angles_TP_mean = np.mean(BP_angles_array_TP, axis=0)

train_loss_DTP_mean = np.mean(train_losses_DTP, axis=0)
test_loss_DTP_mean = np.mean(test_losses_DTP, axis=0)
approx_errors_DTP_mean = np.mean(approx_errors_array_DTP, axis=0)
approx_error_angle_DTP_mean = np.mean(approx_error_angles_array_DTP, axis=0)
inverse_fraction_learning_signal_DTP = np.mean(approx_errors_array_DTP/GN_errors_array_DTP, axis=0)
GN_errors_DTP_mean = np.mean(GN_errors_array_DTP, axis=0)
GN_angles_DTP_mean = np.mean(GN_angles_array_DTP, axis=0)
BP_angles_DTP_mean = np.mean(BP_angles_array_DTP, axis=0)

train_loss_originalTP_mean = np.mean(train_losses_originalTP, axis=0)
test_loss_originalTP_mean = np.mean(test_losses_originalTP, axis=0)
approx_errors_originalTP_mean = np.mean(approx_errors_array_originalTP, axis=0)
approx_error_angle_originalTP_mean = np.mean(approx_error_angles_array_originalTP, axis=0)
inverse_fraction_learning_signal_originalTP = np.mean(approx_errors_array_originalTP/GN_errors_array_originalTP, axis=0)
GN_errors_originalTP_mean = np.mean(GN_errors_array_originalTP, axis=0)
GN_angles_originalTP_mean = np.mean(GN_angles_array_originalTP, axis=0)
BP_angles_originalTP_mean = np.mean(BP_angles_array_originalTP, axis=0)

train_loss_originalDTP_mean = np.mean(train_losses_originalDTP, axis=0)
test_loss_originalDTP_mean = np.mean(test_losses_originalDTP, axis=0)
approx_errors_originalDTP_mean = np.mean(approx_errors_array_originalDTP, axis=0)
approx_error_angle_originalDTP_mean = np.mean(approx_error_angles_array_originalDTP, axis=0)
inverse_fraction_learning_signal_originalDTP = np.mean(approx_errors_array_originalDTP/GN_errors_array_originalDTP, axis=0)
GN_errors_originalDTP_mean = np.mean(GN_errors_array_originalDTP, axis=0)
GN_angles_originalDTP_mean = np.mean(GN_angles_array_originalDTP, axis=0)
BP_angles_originalDTP_mean = np.mean(BP_angles_array_originalDTP, axis=0)

train_loss_BP_mean = np.mean(train_losses_BP, axis=0)
test_loss_BP_mean = np.mean(test_losses_BP, axis=0)

train_loss_fixed_BP_mean = np.mean(train_losses_BP_fixed, axis=0)
test_loss_fixed_BP_mean = np.mean(test_losses_BP_fixed, axis=0)



# ========= PLOTS ===========
fontsize = 12
epochs = np.arange(0, max_epoch+1)
legend1 = ['RTP-AI', 'RDTP-AI', 'original-RTP', 'original-RDTP', 'BP', 'BP-fixed']
legend2 = ['RTP-AI', 'RDTP-AI', 'original-RTP', 'original-RDTP']
# legend1 = ['TP-AI', 'DTP-AI', 'original-TP', 'original-DTP', 'BP', 'BP-fixed']
# legend2 = ['TP-AI', 'DTP-AI', 'original-TP', 'original-DTP']
# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure('training_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, train_loss_TP_mean)
plt.semilogy(epochs, train_loss_DTP_mean)
plt.semilogy(epochs, train_loss_originalTP_mean)
plt.semilogy(epochs, train_loss_originalDTP_mean)
plt.semilogy(epochs, train_loss_BP_mean)
plt.semilogy(epochs, train_loss_fixed_BP_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1, fontsize=fontsize)
plt.show()

fig = plt.figure('test_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, test_loss_TP_mean)
plt.semilogy(epochs, test_loss_DTP_mean)
plt.semilogy(epochs, test_loss_originalTP_mean)
plt.semilogy(epochs, test_loss_originalDTP_mean)
plt.semilogy(epochs, test_loss_BP_mean)
plt.semilogy(epochs, test_loss_fixed_BP_mean)
plt.xlabel(r'epoch', fontsize=fontsize)
plt.ylabel(r'MSE loss', fontsize=fontsize)
plt.legend(legend1, fontsize=fontsize)
plt.show()

fig = plt.figure('approx_error_angles')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(approx_error_angle_TP_mean)
plt.plot(approx_error_angle_DTP_mean)
plt.plot(approx_error_angle_originalTP_mean)
plt.plot(approx_error_angle_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize, loc='upper right')
plt.show()

fig = plt.figure('approx_errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(approx_errors_TP_mean)
plt.semilogy(approx_errors_DTP_mean)
plt.semilogy(approx_errors_originalTP_mean)
plt.semilogy(approx_errors_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{approx}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize, loc='upper right')
plt.show()

fig = plt.figure('learning_signal')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(GN_errors_TP_mean)
plt.semilogy(GN_errors_DTP_mean)
plt.semilogy(GN_errors_originalTP_mean)
plt.semilogy(GN_errors_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{TP}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()

fig = plt.figure('GN angles')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(GN_angles_TP_mean)
plt.plot(GN_angles_DTP_mean)
plt.plot(GN_angles_originalTP_mean)
plt.plot(GN_angles_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize, loc='upper right')
plt.show()

fig = plt.figure('BP angles')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(BP_angles_TP_mean)
plt.plot(BP_angles_DTP_mean)
plt.plot(BP_angles_originalTP_mean)
plt.plot(BP_angles_originalDTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()


fig = plt.figure('learning_signal_ratio_inverse')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.plot(inverse_fraction_learning_signal_TP)
plt.plot(inverse_fraction_learning_signal_DTP)
plt.plot(inverse_fraction_learning_signal_originalTP)
plt.plot(inverse_fraction_learning_signal_originalDTP)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{approx}\|_2/ \|e^{TP}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()