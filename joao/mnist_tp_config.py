import json

mnist_config = {'n1': 1000,
                'n2': 1000,
                'w-init': 0.1,
                'b-init': 1,
                'diff-tp': True,
                'eta1': 0.05,
                'eta2': 0.05,
                'eta3': 0.05,
                'eta-b1': 0.01,
                'eta-b2': 0, # 0.01,
                'lambda': 1,
                'batch-size': 64,
                'test-batch-size': 10000,
                'epochs': 100,
                'cuda': True,
                'seed': None,
                'log-interval': 200,
                'log-file': 'log/mnist_tp_log.txt'}

with open('etc/mnist_tp_config.json', 'w') as outfile:
    json.dump(mnist_config, outfile)
