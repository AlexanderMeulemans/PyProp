import matplotlib.pyplot as plt
import numpy as np
import utils.helper_functions as hf

main_dir = '../logs/gridsearch_invertible_TP_onelayer2/'
test_losses = np.load(main_dir + 'test_losses.npy')
train_losses = np.load(main_dir + 'train_losses.npy')

best_test_results = np.min(test_losses, 3)
succesful_runs = best_test_results != 0

# learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
# distances = [0.1, 0.5, 1.5, 5., 10.]
distances = [8.]
learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
# randomizes = [True, False]
randomizes = [True, False]

print('Train results randomize=True ---------')
train_stats_true = hf.get_stats_gridsearch(train_losses[0,:,:,:], distances, learning_rates)
print('Train results randomize=False ---------')
train_stats_false = hf.get_stats_gridsearch(train_losses[1,:,:,:], distances, learning_rates)
print('Test results randomize=True ---------')
test_stats_true = hf.get_stats_gridsearch(test_losses[0,:,:,:], distances, learning_rates)
print('Test results randomize=False ---------')
test_stats_false = hf.get_stats_gridsearch(test_losses[1,:,:,:], distances, learning_rates)


for i,randomize in enumerate(randomizes):
    for j,distance in enumerate(distances):
        for k, learning_rate in enumerate(learning_rates):
            print('Training combination: randomize={}, '
                  'distance={}, learning_rate={} ...'.format(randomize,
                                                             distance,
                                                             learning_rate))
            if succesful_runs[i,j,k]:
                plt.clf()
                plt.subplot(2,1,1)
                plt.semilogy(train_losses[i,j,k,:])
                plt.subplot(2,1,2)
                plt.semilogy(test_losses[i,j,k,:])
                plt.show()
                print('Best test loss: {}'.format(best_test_results[i,j,k]))
            else:
                print('Failed run')

            input('############')

