import numpy as np
iter = 10000
n = 5
eig_min = np.array([])
eig_min2 = np.array([])

for i in range(iter):
    J = np.random.randn(n,n)
    d = np.random.randn(n)
    d2 = np.square(d)
    D2 = np.diag(d2)
    J_inv = np.linalg.inv(J)
    A = np.matmul(J,np.matmul(D2,J_inv))
    l,v = np.linalg.eig(A)
    l_2, v_2 = np.linalg.eig(J)
    eig_min = np.append(eig_min, np.min(np.real(l)))
    eig_min2 = np.append(eig_min2, np.min(np.real(l_2)))

print(np.min(eig_min))
print(np.min(eig_min2))