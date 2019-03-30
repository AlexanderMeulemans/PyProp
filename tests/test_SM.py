import torch
import numpy as np
import random
import matplotlib.pyplot as plt

seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

plt.close()
errors = np.array([])
alphas = np.array([])
betas = np.array([])
errors_inv = np.array([])
for i in range(4,300,4):
    A = 5*torch.randn(5,5)
    u = 0.5*torch.randn(5,1)
    v = 0.5*torch.randn(5,1)
    A_inv = torch.inverse(A)
    d = torch.matmul(torch.transpose(v, -1, -2),
                             torch.matmul(A_inv, u))

    epsilon = 1e-2
    alpha = (torch.sign(1 + d) * epsilon - 1) / d
    beta = 1 / (1 + (alpha - 1) * d)


    At = A + beta*torch.matmul(u, torch.transpose(v, -1, -2))
    At_inv = torch.inverse(At)



    denominator = torch.sign(1+d)*epsilon
    numerator = torch.matmul(torch.matmul(A_inv, u),
                                     torch.matmul(torch.transpose(v, -1, -2),
                                                  A_inv))
    At_inv_SM = A_inv - torch.div(numerator, denominator)

    error = torch.norm(torch.eye(5)-At_inv_SM*At)
    error_inv = torch.norm(At_inv-At_inv_SM)

    errors = np.append(errors, error.numpy())
    alphas = np.append(alphas,alpha.numpy())
    betas = np.append(betas,beta.numpy())
    errors_inv = np.append(errors_inv, error_inv.numpy())

plt.figure()
plt.semilogy(errors)
plt.title('errors')
plt.show()

plt.figure()
plt.semilogy(errors_inv)
plt.title('errors_inv')
plt.show()

plt.figure()
plt.semilogy(betas)
plt.title('betas')
plt.show()

plt.figure()
plt.semilogy(alphas)
plt.title('alpha')
plt.show()

# print('error: {}'.format(error))
# print('beta: {}'.format(beta))
# print('alpha: {}'.format(alpha))
# print(At_inv)
# print(At_inv_SM)


