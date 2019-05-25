import numpy as np
import matplotlib.pyplot as plt

iterations = 1000
n=5
distances = np.zeros(iterations)

for i in range(iterations):
    a = np.random.randn(n,n)
    b = np.random.randn(n,n)
    c = a-b
    distances[i] = np.linalg.norm(c)

plt.hist(distances,bins=30)
plt.show()