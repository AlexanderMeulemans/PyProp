import numpy as np
import matplotlib.pyplot as plt


log_dir_main = '../logs/final_combined_toy_example_invertibleTP_save/'
log_dir_save = '../logs/final_combined_toy_example_unequal_layers_save/'
max_epoch = 60

indices = [0,1,2,4,5,6,7,8,9,10,11,12,13,14]

# ====== load results =======
train_losses_TP = np.load(log_dir_main + 'train_losses_TP.npy')
test_losses_TP = np.load(log_dir_main + 'test_losses_TP.npy')
approx_error_angles_array_TP=np.load(log_dir_main + 'approx_error_angles_array_TP.npy')
approx_errors_array_TP=np.load(log_dir_main + 'approx_errors_array_TP.npy')

train_losses_TPrandom=np.load(log_dir_main + 'train_losses_TPrandom.npy')
test_losses_TPrandom=np.load(log_dir_main + 'test_losses_TPrandom.npy')
approx_error_angles_array_TPrandom=np.load(log_dir_main + 'approx_error_angles_array_TPrandom.npy')
approx_errors_array_TPrandom=np.load(log_dir_main + 'approx_errors_array_TPrandom.npy')

train_losses_MTP=np.load(log_dir_main + 'train_losses_MTP.npy')
test_losses_MTP=np.load(log_dir_main + 'test_losses_MTP.npy')
approx_error_angles_array_MTP=np.load(log_dir_main + 'approx_error_angles_array_MTP.npy')
approx_errors_array_MTP=np.load(log_dir_main + 'approx_errors_array_MTP.npy')

train_losses_BP=np.load(log_dir_main + 'train_losses_BP.npy')
test_losses_BP=np.load(log_dir_main + 'test_losses_BP.npy')

train_losses_BP_fixed=np.load(log_dir_main + 'train_losses_BP_fixed.npy')
test_losses_BP_fixed=np.load(log_dir_main + 'test_losses_BP_fixed.npy')

# train_losses_TP = train_losses_TP[indices,:]
# test_losses_TP=test_losses_TP[indices,:]
# approx_error_angles_array_TP=approx_error_angles_array_TP[indices,:]
# approx_errors_array_TP=approx_errors_array_TP[indices,:]
#
# train_losses_TPrandom=train_losses_TPrandom[indices,:]
# test_losses_TPrandom=test_losses_TPrandom[indices,:]
# approx_error_angles_array_TPrandom=approx_error_angles_array_TPrandom[indices,:]
# approx_errors_array_TPrandom=approx_errors_array_TPrandom[indices,:]
#
# train_losses_MTP=train_losses_MTP[indices,:]
# test_losses_MTP=test_losses_MTP[indices,:]
# approx_error_angles_array_MTP=approx_error_angles_array_MTP[indices,:]
# approx_errors_array_MTP=approx_errors_array_MTP[indices,:]
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
#
# np.save(log_dir_save + 'train_losses_TPrandom.npy', train_losses_TPrandom)
# np.save(log_dir_save + 'test_losses_TPrandom.npy', test_losses_TPrandom)
# np.save(log_dir_save + 'approx_error_angles_array_TPrandom.npy', approx_error_angles_array_TPrandom)
# np.save(log_dir_save + 'approx_errors_array_TPrandom.npy', approx_errors_array_TPrandom)
#
# np.save(log_dir_save + 'train_losses_MTP.npy', train_losses_MTP)
# np.save(log_dir_save + 'test_losses_MTP.npy', test_losses_MTP)
# np.save(log_dir_save + 'approx_error_angles_array_MTP.npy', approx_error_angles_array_MTP)
# np.save(log_dir_save + 'approx_errors_array_MTP.npy', approx_errors_array_MTP)
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

train_loss_TPrandom_mean = np.mean(train_losses_TPrandom, axis=0)
test_loss_TPrandom_mean = np.mean(test_losses_TPrandom, axis=0)
approx_errors_TPrandom_mean = np.mean(approx_errors_array_TPrandom, axis=0)
approx_error_angle_TPrandom_mean = np.mean(approx_error_angles_array_TPrandom, axis=0)

train_loss_MTP_mean = np.mean(train_losses_MTP, axis=0)
test_loss_MTP_mean = np.mean(test_losses_MTP, axis=0)
approx_errors_MTP_mean = np.mean(approx_errors_array_MTP, axis=0)
approx_error_angle_MTP_mean = np.mean(approx_error_angles_array_MTP, axis=0)

train_loss_BP_mean = np.mean(train_losses_BP, axis=0)
test_loss_BP_mean = np.mean(test_losses_BP, axis=0)

train_loss_fixed_BP_mean = np.mean(train_losses_BP_fixed, axis=0)
test_loss_fixed_BP_mean = np.mean(test_losses_BP_fixed, axis=0)


# ========== Smooth results =========
def smooth(y, box_pts=8):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

approx_errors_MTP_mean = smooth(approx_errors_MTP_mean)
approx_error_angle_MTP_mean = smooth(approx_error_angle_MTP_mean)
approx_errors_TP_mean = smooth(approx_errors_TP_mean)
approx_error_angle_TP_mean = smooth(approx_error_angle_TP_mean)
approx_errors_TPrandom_mean = smooth(approx_errors_TPrandom_mean)
approx_error_angle_TPrandom_mean = smooth(approx_error_angle_TPrandom_mean)



# ========= PLOTS ===========
fontsize = 14
epochs = np.arange(0, max_epoch+1)
legend1 = ['TP-EI', 'RTP-EI', 'RMTP-EI', 'BP', 'BP-fixed']
legend2 = ['TP-EI', 'RTP-EI', 'RMTP-EI']
# Set plot style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure('training_loss')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(epochs, train_loss_TP_mean)
plt.semilogy(epochs, train_loss_TPrandom_mean)
plt.semilogy(epochs, train_loss_MTP_mean)
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
plt.semilogy(epochs, test_loss_TPrandom_mean)
plt.semilogy(epochs, test_loss_MTP_mean)
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
plt.plot(approx_error_angle_TPrandom_mean)
plt.plot(approx_error_angle_MTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\cos(\alpha)$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize)
plt.show()

fig = plt.figure('approx_errors')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.semilogy(approx_errors_TP_mean)
plt.semilogy(approx_errors_TPrandom_mean)
plt.semilogy(approx_errors_MTP_mean)
plt.xlabel(r'mini-batch', fontsize=fontsize)
plt.ylabel(r'$\|e^{approx}\|_2$', fontsize=fontsize)
plt.legend(legend2, fontsize=fontsize, loc='upper right')
plt.show()
