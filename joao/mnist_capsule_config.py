import json
import os

mnist_config = {'n1': 784,
                'n2': 784,
                'w-init': 0.1,
                'b-init': 1,
                'eta1': 0.05,
                'eta2': 0.05,
                'batch-size': 64,
                'test-batch-size': 10000,
                'epochs': 100,
                'cuda': True,
                'seed': 1,
                'log-interval': 10,
                'log-file': 'log/mnist_capsule_log.txt'}


path = 'etc/mnist_capsule_config.json'
exists = os.path.isfile(path)
if exists:
    with open(path, 'w') as outfile:
        json.dump(mnist_config, outfile)
else:
    open(path, 'w').close()
    with open(path, 'w') as outfile:
        json.dump(mnist_config, outfile)