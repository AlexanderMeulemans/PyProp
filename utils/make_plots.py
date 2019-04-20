import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BP_loss = pd.read_csv('../figure_data/run_BP,tag_test_loss.csv')
TP_loss = pd.read_csv('../figure_data/run_TP,tag_test_loss.csv')
shallow_loss = pd.read_csv('../figure_data/run_shallow,tag_test_loss.csv')

length = 14
epochsBP = range(1,length+1)
epochsTP = range(1,length+1)
epochsSh = range(1,length+1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure()

plt.semilogy(epochsBP, BP_loss['Value'][0:length])
plt.semilogy(epochsTP, TP_loss['Value'][0:length])
plt.semilogy(epochsSh, shallow_loss['Value'][0:length])
plt.title('Test loss nonlinear regression toy example')
plt.xlabel('epoch')
plt.ylabel(r'$L_2$ loss')
plt.legend(['Backpropagation', 'Target propagation', 'Fixed hidden layer'])
plt.show()