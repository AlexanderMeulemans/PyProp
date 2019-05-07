import numpy as np
import matplotlib.pyplot as plt


if True:
    block_size = 5
    nb_blocks = 5
    iterations = 10000
    eig_max = np.array([])
    mu = 0.5 * np.random.rand(1)
    for j in range(iterations):

        J_bar = np.zeros((block_size*nb_blocks, block_size*nb_blocks))
        for i in range(nb_blocks-1):
            s = 5 * np.random.rand(1)
            # s = 5000
            J_bar[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = -np.eye(block_size)
            J = 100000*np.random.rand(block_size, block_size)
            J2 = 1/s*np.random.randn(block_size, block_size)
            J_bar[(i+1)*block_size:(i+2)*block_size, i*block_size:(i+1)*block_size] = (1-mu)*J
            J_bar[i * block_size:(i + 1) * block_size,
            (i + 1) * block_size:(i + 2) * block_size,] = mu*np.linalg.inv(J)
            # J_bar[i * block_size:(i + 1) * block_size,
            # (i + 1) * block_size:(i + 2) * block_size,] = mu*J2

        J_bar[(nb_blocks-1)*block_size: nb_blocks*block_size,
        (nb_blocks-1)*block_size: nb_blocks*block_size]  = -np.eye(block_size)

        l,v = np.linalg.eig(J_bar)
        eig_max = np.append(eig_max, np.max(np.real(l)))

    print(np.max(eig_max))
    plt.hist(eig_max)
    plt.show()

if False:
    sigma = np.linspace(0,10,1000)
    l = 0.3
    f = l*sigma**(-1) + (1-l)*sigma

    plt.figure()
    plt.plot(sigma, f)
    plt.show()