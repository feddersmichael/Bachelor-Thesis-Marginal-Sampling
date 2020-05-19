import scipy.integrate as integrate
import math
import numpy as np
import scipy.special as special
import scipy


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
    prefactor = 1 / ((2 * math.pi * s) ** (N / 2))
    integr = 0
    for k in range(N):
        integr += (y[k] - c - h[k]) ** 2
    integr = math.expm1(integr / (2 * s))
    return prefactor * integr


def analytical(y, h, N):
    product_1 = N ** (N / 2 - 2)
    product_2 = 2 * np.pi ** ((N - 1) / 2)
    sum_1 = np.sum((h - y) ** 2)
    sum_2 = np.sum(y - h) ** 2
    product_3 = (sum_1 - sum_2) ** (-(N - 3) / 2)
    return product_1 * product_3 * special.gamma((N - 3) / 2) / product_2


def main():
    c = 0
    N = 10
    theta = 0
    sigma = 1
    h = np.zeros(N)
    y = np.zeros(N)
    for n in range(N):
        h[n] = h_func(n + 1, theta)
        y[n] = y_measured(n + 1, c, theta, sigma)
    solution = analytical(y, h, N)
    result = scipy.integrate.dblquad(integrand, 0.1, 3, lambda x: -3, lambda y: 3, args=(y, h, N))
    print(result)


main()
