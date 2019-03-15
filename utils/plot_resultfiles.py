"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


results = pd.read_csv('resultfile.csv')
results_shallow = pd.read_csv(('resultfile_shallow.csv'))
# results_combined = results.join(results_shallow)

results_train = results.as_matrix(['Train_loss'])
results_test = results.as_matrix(['Test_loss'])
results_shallow_train = results_shallow.as_matrix(['Train_loss'])
results_shallow_test = results_shallow.as_matrix(['Test_loss'])

plt.figure()
plt.plot(results_train)
plt.plot(results_test)
plt.plot(results_shallow_train)
plt.plot(results_shallow_test)
plt.xlabel('epoch')
plt.ylabel('L2 loss')
plt.legend(['TP train', 'TP test', 'shallow train', 'shallow test'])


