import matplotlib.pyplot as plt
import numpy as np
import utils.helper_functions as hf

main_dir = '../logs/gridsearch_normal_TP_2layers/'
test_losses = np.load(main_dir + 'test_losses.npy')
train_losses = np.load(main_dir + 'train_losses.npy')

best_test_results = np.min(test_losses, 3)
succesful_runs = best_test_results != 0

learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
distances = [0.1, 0.5, 1.5, 5., 10.]
randomizes = [True, False]
backward_weight_decays = [0.1, 0.01, 0.0]

print('Train results randomize=True ---------')
train_stats_true = hf.get_stats_gridsearch2(train_losses[0,:,:,:,:], distances, learning_rates,
                         backward_weight_decays)
print('Train results randomize=False ---------')
train_stats_false = hf.get_stats_gridsearch2(train_losses[1,:,:,:,:], distances, learning_rates,
                         backward_weight_decays)
print('Test results randomize=True ---------')
test_stats_true = hf.get_stats_gridsearch2(test_losses[0,:,:,:,:], distances, learning_rates,
                         backward_weight_decays)
print('Test results randomize=False ---------')
test_stats_false = hf.get_stats_gridsearch2(test_losses[1,:,:,:,:], distances, learning_rates,
                         backward_weight_decays)


# for i,randomize in enumerate(randomizes):
#     for j,distance in enumerate(distances):
#         for k, learning_rate in enumerate(learning_rates):
for i, randomize in enumerate(randomizes):
    for j, distance in enumerate(distances):
        for k, learning_rate in enumerate(learning_rates):
            for l, weight_decay in enumerate(backward_weight_decays):
                print('Training combination: randomize={}, '
                      'distance={}, learning_rate={}, '
                      'weight_decay={}...'.format(randomize,
                                                  distance,
                                                  learning_rate,
                                                  weight_decay))
                if succesful_runs[i, j, k, l]:
                    plt.clf()
                    plt.subplot(2, 1, 1)
                    plt.semilogy(train_losses[i, j, k,l, :])
                    plt.subplot(2, 1, 2)
                    plt.semilogy(test_losses[i, j, k,l, :])
                    plt.show()
                    print(
                        'Best test loss: {}'.format(best_test_results[i, j, k,l]))
                else:
                    print('Failed run')

                input('############')

