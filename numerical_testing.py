import scipy.integrate as integrate
import math
import numpy as np


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


def main():
    c = 0
    h = [h_func(n, 0) for n in range(1, 10)]
    y = [y_measured(n, c, 0, 1) for n in range(1, 10)]

# ans, err = scipy.integrate.dblquad(integrand, 0, 20, lambda x: -10, lambda x: 10)
