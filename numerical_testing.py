import scipy.integrate as integrate
import math
import numpy as np
import scipy.special as special
import scipy
import matplotlib.pyplot as plt


def h_func(t, theta):
    """observation function"""
    return t


def y_orig(c, t, theta):
    """observation function with a transformation factor"""
    return c + h_func(t, theta)


def y_measured(t, c, theta, sigma):
    """Adds noise to data y.

    We get the data y as an input and simulate
    random noise which represents the error.
    The measured data y_tilde gets returned.
    """
    return np.random.normal(y_orig(c, t, theta), sigma)


def integrand(c, s, y, h, N):
    """integrand of our marginalisation

    c and s = sigma^2 are the integration variables.
    y and h are represnting the measured data at time points,
    so they com in lists of length N, which is the amount
    of time-points."""
    prefactor = (2 * math.pi * s) ** (-N / 2)
    integr = 0
    for k in range(N):  # TODO numpy writing
        integr += (y[k] - c - h[k]) ** 2
    integr = math.exp(integr / (-2 * s))  # TODO numpy exponential
    return prefactor * integr


def analytical(y, h, N):
    """analytical solution for the integral"""
    product_1 = N ** (N / 2 - 2)
    product_2 = 2 * np.pi ** ((N - 1) / 2)
    sum_1 = N * np.sum((h - y) ** 2)
    sum_2 = np.sum(y - h) ** 2
    product_3 = (sum_1 - sum_2) ** ((N - 3) / -2)
    product_4 = special.gamma((N - 3) / 2)
    return product_1 * product_3 * product_4 / product_2


def main():
    c = np.linspace(-10, 10, 10)
    N = 10
    theta = 0
    sigma = np.linspace(0.02, 2, 10)  # sigma in the sense of sigma^2
    value = np.empty((c.size, sigma.size))
    #for i in range(c.size):
    #    for j in range(sigma.size):
    #        value[i, j] = result(c[i], sigma[j], theta, N)

    #fig = plt.figure()
    #s = fig.add_subplot(1, 1, 1, xlabel='$\\sigma$', ylabel='c')
    im = plt.imshow(
        value,
        extent=(sigma[0], sigma[-1], c[0], c[-1]))
    #fig.colorbar(im)
    #fig.savefig('heatmap.png')
    plt.show()

    # solution = analytical(y, h, N)
    # result = integrate.dblquad(integrand, 0, 1000, lambda x: -5+c, lambda x: 5+c, args=(y, h, N))


def result(c, sigma, theta, N):
    mean_1 = 0
    mean_2 = 0
    # for z in range(10):
    h = np.zeros(N)
    y = np.zeros(N)
    for n in range(N):
        h[n] = h_func(n + 1, theta)
        y[n] = y_measured(n + 1, c, theta, sigma)
    mean_1 = integrate.dblquad(integrand, 0, min(10, 5 * sigma), lambda x: -5 + c, lambda x: 5 + c, args=(y, h, N))[0]
    mean_2 = analytical(y, h, N)
    # mean_1 = mean_1/10
    # mean_2 = mean_2/10
    return abs(mean_1 - mean_2) / mean_2


main()
