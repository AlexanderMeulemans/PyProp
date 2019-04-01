import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import utils.helper_functions as hf

seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# set plot layout to latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# User variables
plot_s_d = False
plot_a_b = False
plot_s_min_robust = True
plot_s_eps = False
matrix_size = 5
threshold = 0.1




# experiments



if plot_s_d:
    s_mins = np.array([])
    s_maxs = np.array([])
    ds = np.array([])
    s_min_min = 1000
    A_min = None
    u_min = None
    v_min = None
    for i in range(5000):
        n = matrix_size
        A = 5 * hf.get_invertible_random_matrix(n,n)
        u = 0.5 * torch.randn(n, 1)
        v = 0.5 * torch.randn(n, 1)
        A_inv = torch.inverse(A)
        d = torch.matmul(torch.transpose(v, -1, -2),
                         torch.matmul(A_inv, u))
        At = A + torch.matmul(u, torch.transpose(v, -1, -2))
        U, S, V = torch.svd(At)
        s_max = S[0]
        s_min = S[-1]
        s_mins = np.append(s_mins, s_min)
        s_maxs = np.append(s_maxs, s_max)
        ds = np.append(ds, torch.abs(1+d))

        # save A, u and v with At closest to singularity for later use
        if s_min < s_min_min:
            A_min = A
            u_min = u
            v_min = v
            s_min_min = s_min
    plt.figure()
    plt.loglog(ds,s_mins,'*')
    plt.title(r'$s_{min}$ vs $d$')
    plt.xlabel(r'$|d+1|$')
    plt.ylabel(r'$s_{min}$')
    plt.show()

    # plt.figure()
    # plt.loglog(ds, s_maxs, '*')
    # plt.title('d versus s_max')
    # plt.show()
    np.save('A_min.npy', A_min)
    np.save('u_min.npy', u_min)
    np.save('v_min.npy', v_min)


if plot_a_b:
    n = 5000
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5]
    legend = [str(eps) for eps in epsilons]
    alphas = np.zeros((n, len(epsilons)))
    betas = np.zeros((n, len(epsilons)))
    d = 0.1 * np.random.randn(5000) - 1
    for i, epsilon in enumerate(epsilons):
        alpha = (np.sign(1 + d) * epsilon - 1) / d
        beta = 1 / (1 + (alpha - 1) * d)
        alphas[:,i] = alpha
        betas[:,i] = beta

    plt.figure()
    plt.loglog(np.abs(d+1), alphas, '.')
    plt.legend(legend)
    plt.title(r'$\alpha$')
    plt.xlabel(r'$|d+1|$')
    plt.ylabel(r'$\alpha$')
    plt.show()

    plt.figure()
    plt.loglog(np.abs(d + 1), betas, '.')
    plt.legend(legend)
    plt.title(r'$\beta$')
    plt.xlabel(r'$|d+1|$')
    plt.ylabel(r'$\beta$')
    plt.show()


if plot_s_min_robust:
    s_mins = np.array([])
    s_maxs = np.array([])
    ds = np.array([])
    betas = np.array([])
    alphas = np.array([])
    errors = np.array([])
    errors_robust = np.array([])
    errors_notrobust = np.array([])
    errors_total = np.array([])

    n = matrix_size
    epsilon = threshold

    for i in range(20000):
        A = 5 * hf.get_invertible_random_matrix(n,n)
        u = 0.5 * torch.randn(n, 1)
        v = 0.5 * torch.randn(n, 1)
        A_inv = torch.inverse(A)
        d = torch.matmul(torch.transpose(v, -1, -2),
                         torch.matmul(A_inv, u))
        numerator = torch.matmul(torch.matmul(A_inv, u),
                                 torch.matmul(torch.transpose(v, -1, -2),
                                              A_inv))
        if abs(1+d)<epsilon:
            alpha = (epsilon - 1) / d
            beta = 1 / (epsilon - d)
            alphas = np.append(alphas, alpha)
            betas = np.append(betas, beta)
            At = A + beta * torch.matmul(u, torch.transpose(v, -1, -2))
            denominator = epsilon
            At_inv_SM = A_inv - torch.div(numerator, denominator)
            error = torch.norm(torch.eye(n) - At_inv_SM * At)
            errors_robust = np.append(errors_robust, error)
            errors_total = np.append(errors_total, error)
            At_inv_SM_notrobust = A_inv - torch.div(numerator, 1+d)
            At_notrobust = A + torch.matmul(u, torch.transpose(v, -1, -2))
            U, S, V = torch.svd(At)
            s_max = S[0]
            s_min = S[-1]
            s_mins = np.append(s_mins, s_min)
            error = torch.norm(torch.eye(n) - At_inv_SM_notrobust * At_notrobust)
            errors_notrobust = np.append(errors_notrobust, error)

        else:
            At = A + torch.matmul(u, torch.transpose(v, -1, -2))
            denominator = 1+d
            At_inv_SM = A_inv - torch.div(numerator, denominator)
            error = torch.norm(torch.eye(n) - At_inv_SM * At)
            errors = np.append(errors, error)
            errors_total = np.append(errors_total, error)
            U, S, V = torch.svd(At)
            s_max = S[0]
            s_min = S[-1]
            s_mins = np.append(s_mins, s_min)
            betas = np.append(betas, 1.)




        ds = np.append(ds, torch.abs(1+d))

    errors_robust = errors_robust/(n)
    errors_notrobust = errors_notrobust/(n)
    errors = errors/(n)
    errors_total = errors_total/(n)

    plt.figure()
    plt.loglog(ds,s_mins,'*')
    plt.title(r'$s_{min}$ vs $d$')
    plt.xlabel(r'$|d+1|$')
    plt.ylabel(r'$s_{min}$')
    plt.show()

    plt.figure()
    plt.hist(betas, bins=30)
    plt.title(r'$\beta$')
    plt.show()

    # plt.figure()
    # plt.hist(alphas, bins=30)
    # plt.title(r'$\alpha$')
    # plt.show()

    plt.figure()
    plt.hist(errors, bins=30)
    plt.title('errors normal SM')
    plt.show()

    plt.figure()
    plt.hist(errors_total * s_mins*np.sqrt(n), bins=50)
    plt.title('inverse error times $s_{min}$')
    plt.show()

    plt.figure()
    plt.hist(errors_robust, bins=30)
    plt.title('errors robust SM')
    plt.show()

    plt.figure()
    plt.hist(errors_notrobust, bins=30)
    plt.title('errors not robust SM')
    plt.show()

    plt.figure()
    plt.hist(errors_total, bins=30)
    plt.title('total errors')

    plt.figure()
    plt.loglog(ds, errors_total, '*')
    plt.title('errors vs $d$')
    plt.show()



if plot_s_eps:
    s_mins = np.array([])
    errors = np.array([])
    n = matrix_size
    epsilons = np.logspace(-4, 1, 100)
    A = torch.from_numpy(np.load('A_min.npy'))
    u = torch.from_numpy(np.load('u_min.npy'))
    v = torch.from_numpy(np.load('v_min.npy'))
    A_inv = torch.inverse(A)
    d = torch.matmul(torch.transpose(v, -1, -2),
                     torch.matmul(A_inv, u))
    print('d: {}'.format(d))
    for epsilon in epsilons:
        beta = 1/(np.sign(1+d)*epsilon -d)
        At = A + beta * torch.matmul(u, torch.transpose(v, -1, -2))
        U, S, V = torch.svd(At)
        s_max = S[0]
        s_min = S[-1]
        s_mins = np.append(s_mins, s_min)
        numerator = torch.matmul(torch.matmul(A_inv, u),
                                 torch.matmul(torch.transpose(v, -1, -2),
                                              A_inv))
        denominator = torch.sign(1 + d) * epsilon
        At_inv_SM = A_inv - torch.div(numerator, denominator)
        error = torch.norm(torch.eye(n) - At_inv_SM * At)
        errors = np.append(errors, error)

    plt.figure()
    plt.loglog(epsilons, s_mins,'*')
    plt.title(r'$s_{min}$ vs  $\epsilon$')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$s_{min}$')
    plt.show()

    plt.figure()
    plt.loglog(epsilons, errors, '*')
    plt.title(r'error vs  $\epsilon$')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'error')
    plt.show()








