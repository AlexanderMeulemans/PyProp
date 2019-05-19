import matplotlib.pyplot as plt
import numpy as np

test_losses = np.load('../logs/test_losses.npy')
train_losses = np.load('../logs/train_losses.npy')

best_test_results = np.min(test_losses, 3)
succesful_runs = best_test_results != 0

learning_rates = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
distances = [0.1, 0.5, 1.5, 5., 10.]
randomizes = [True, False]

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
                plt.plot(train_losses[i,j,k,:])
                plt.subplot(2,1,2)
                plt.plot(test_losses[i,j,k,:])
                plt.show()
                print('Best test loss: {}'.format(best_test_results[i,j,k]))
            else:
                print('Failed run')

            input('############')

