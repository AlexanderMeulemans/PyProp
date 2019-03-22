import numpy as np
from matplotlib import pyplot as plt

iterations = 1000
bins = int(iterations/10)
s_max = np.empty((iterations,1))
s_min = np.empty((iterations,1))

for i in range(iterations):
    a = np.random.randn(100,100)
    U,S,V = np.linalg.svd(a)
    s_max[i] = S[0]
    s_min[i] = S[-1]

plt.figure()
plt.hist(s_max, bins=bins)
plt.show()

plt.figure()
plt.hist(s_min, bins=bins)
plt.show()