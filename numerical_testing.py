import scipy.integrate as integrate
import math
import numpy as np
import scipy.special as special
import scipy
import matplotlib.pyplot as plt


def h_func_srm(t, theta):
    """observation function"""
    return t


def X_2(t):
    return t


def h_func_crm(t, theta):
    """observation function"""
    return X_2(t)


def y_func_offset(c, t, theta):
    """observation function with a transformation factor"""
    return c + h_func_srm(t, theta)


def y_meas_add_gaus(t, c, theta, sigma):
    """Adds noise to data y.

    We get the data y as an input and simulate
    random noise which represents the error.
    The measured data y_tilde gets returned.
    """
    return np.random.normal(y_func_offset(c, t, theta), sigma)


def integrand_flat_prior(c, s, y, h, N):
    """integrand of our marginalisation for flat prior

    c and s = sigma^2 are the integration variables.
    y and h are representing the measured data at time points,
    so they are lists of length N, which is the amount
    of time-points."""
    prefactor = (2 * np.pi * s) ** (-N / 2)
    integr = 0
    for k in range(N):  # TODO numpy writing
        integr += (y[k] - c - h[k]) ** 2
    integr = np.exp(integr / (-2 * s))
    return prefactor * integr


def integrand_normal_gamme(c, l, y, h, mu, kappa, alpha, beta, N):
    """integrand of our marginalization for normal-gamma prior

    l = lambda = 1/s^2 and c are our integration variables.
    mu, kappa, alpha and beta are specifying our normal-gamma distribution
    y and h are the measured data points of length N
    N equals the amount of time points"""
    prefactor_1 = beta ** alpha * np.sqrt(kappa) * np.e ** (2 * beta)
    prefactor_2 = scipy.special.gamma(alpha) * (2 * np.pi) ** ((N + 1) / 2)
    prefactor_3 = l ** ((N + 2 * alpha - 1) / 2)
    integr = 0
    for k in range(N):
        integr += (y[k] - h[k] - c) ** 2
    integr += kappa * (c - mu) ** 2
    integr = np.exp(-integr * l / 2)
    return integr * (prefactor_1 / prefactor_2) * prefactor_3


def analytical_flat_prior(y, h, N):
    """analytical solution for the integral"""
    product_1 = N ** (N / 2 - 2)
    product_2 = 2 * np.pi ** ((N - 1) / 2)
    sum_1 = N * np.sum((h - y) ** 2)
    sum_2 = np.sum(y - h) ** 2
    product_3 = (sum_1 - sum_2) ** ((N - 3) / -2)
    product_4 = special.gamma((N - 3) / 2)
    return product_1 * product_3 * product_4 / product_2


def analytical_normal_gamma_prior(y, h, mu, kappa, alpha, beta, N):
    C_1 = 0
    C_2 = 0
    for k in range(N):
        temp = y[k] - h[k]
        C_1 += temp**2
        C_2 += temp
    C_1 += kappa*mu**2
    C_2 = (C_2 + kappa*mu)**2
    C = C_1/2 - C_2/(2*(N + kappa))
    prefactor_1 = beta**alpha * np.sqrt(kappa) * np.e**(2 *beta)
    prefactor_2 = scipy.special.gamma(alpha) * (2* np.pi)**(N/2) * (N + kappa) * C**((N + 2*alpha)/2)
    return prefactor_1/prefactor_2 * scipy.special.gamma((N + 2*alpha)/2)


def result(c, sigma, theta, N):
    mean_1 = 0
    mean_2 = 0
    # for z in range(10):
    h = np.zeros(N)
    y = np.zeros(N)
    for n in range(N):
        h[n] = h_func_srm(n + 1, theta)
        y[n] = y_meas_add_gaus(n + 1, c, theta, sigma)
    mean_1 = \
        integrate.dblquad(integrand_flat_prior, 0, min(10, 5 * sigma), lambda x: -5 + c, lambda x: 5 + c,
                          args=(y, h, N))[0]
    mean_2 = analytical_flat_prior(y, h, N)
    # mean_1 = mean_1/10
    # mean_2 = mean_2/10
    return abs(mean_1 - mean_2) / mean_2


def sample_regression():
    c = np.linspace(-10, 10, 20)
    N = 10
    theta = 0
    sigma = np.linspace(0.02, 2, 20)  # sigma in the sense of sigma^2
    value = np.empty((c.size, sigma.size))
    for i in range(c.size):
        for j in range(sigma.size):
            value[i, j] = result(c[i], sigma[j], theta, N)

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='$\\sigma$', ylabel='c')
    im = plt.imshow(
        value, origin='lower', interpolation='bilinear',
        aspect='equal')  # extent=(sigma[0], sigma[-1], c[0], c[-1])
    fig.colorbar(im)
    fig.savefig('heatmap.png')
    plt.show()

    # solution = analytical(y, h, N)
    # result = integrate.dblquad(integrand, 0, 1000, lambda x: -5+c, lambda x: 5+c, args=(y, h, N))


def conversion_reaction():
    c = np.linspace(-10, 10, 20)
    N = 10
    theta = 0
    sigma = 0.2  # sigma in the sense of sigma^2
    value = np.empty(c.size)
    for i in range(c.size):
        value[i] = result(c[i], sigma, theta, N)
    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='$\\sigma$', ylabel='c')
    im = plt.imshow(
        value, origin='lower', interpolation='bilinear',
        aspect='equal')  # extent=(sigma[0], sigma[-1], c[0], c[-1])
    fig.colorbar(im)
    fig.savefig('heatmap.png')
    plt.show()


def main():
    sample_regression()


main()
