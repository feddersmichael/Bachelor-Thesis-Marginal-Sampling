"""
Numerical testing of the analytically derived integral
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.special import erf
import seaborn as sns

# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({'font.size': 17})

d = os.getcwd()

mu = 0
tau = 1
observable = np.array(range(1, 11))
N = 10


def integrand_offset(offset, shape, measurement):
    """

    :param shape:
    :param offset:
    :param measurement:
    :return:
    """
    result = np.exp(- (tau / 2) * (offset - mu) ** 2 - np.sum(abs(measurement - offset - observable)) / shape) \
             * np.sqrt(tau / (2 * np.pi))
    return result


def substitution_function(x, i, shape):
    return (x - ((N - 2 * i) / (tau * shape) + mu)) / np.sqrt(2 / tau)


def analytical_normal_gamma_prior(measurement, shape):
    data = measurement

    # simulate model
    simulation = observable

    # evaluate standard log likelihood
    # We sort the difference of data and simulation in increasing order
    res = data - simulation
    b_vector = np.sort(res)
    bounds = np.append(np.append(-np.inf, b_vector), np.inf)

    # We initiate l_value with for i = 0
    l_value = - np.sum(b_vector[:])

    # First comes the infinite interval [-\infty, b_1]

    f_2 = substitution_function(bounds[1], 1, shape)
    if f_2 <= 0:
        k_value = 1 - erf(-f_2)
    else:
        k_value = 1 + erf(f_2)

    marginal_posterior = np.exp(N ** 2 / (2 * tau * shape ** 2) + (mu * N + l_value) / shape) * k_value

    # Now we go through all finite intervals
    for i in range(1, N):
        l_value += 2 * bounds[i]
        f_1 = substitution_function(bounds[i], i, shape)
        f_2 = substitution_function(bounds[i + 1], i, shape)
        if f_2 <= 0:
            k_value = erf(-f_1) - erf(-f_2)
        elif 0 < f_1:
            k_value = erf(f_2) - erf(f_1)
        else:
            k_value = erf(-f_1) + erf(f_2)

        marginal_posterior += np.exp(
            (N - 2 * i) ** 2 / (2 * tau * shape ** 2) + (mu * (N - 2 * i) + l_value) / shape) * k_value

    # Last case interval [b_N, +\infty]

    l_value += 2 * bounds[N]
    f_1 = substitution_function(bounds[N], N, shape)
    if f_1 <= 0:
        k_value = erf(-f_1) + 1
    else:
        k_value = 1 - erf(f_1)

    marginal_posterior += np.exp(N ** 2 / (2 * tau * shape ** 2) + (-mu * N + l_value) / shape) * k_value

    log_marginal_posterior = np.log(marginal_posterior)
    log_marginal_posterior += -(N + 1) * (np.log(2)) - N * np.log(shape)

    return -log_marginal_posterior


def relative_error():
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    Generator = np.random.default_rng()
    df = [np.zeros(21 * 20), np.zeros(21 * 20), np.zeros(21 * 20)]
    for n, value in enumerate(np.linspace(-10, 10, 21)):
        df[0][n * 20:(n + 1) * 20] = value
    shape_values = [0.5 + n / 10 for n in range(1, 21)]
    for n, value in enumerate(df[1]):
        df[1][n] = shape_values[n % 20]
    measurement = [np.ones(10)]
    for i, value in enumerate(df[2]):
        for n in range(1, 11):
            measurement[0][n - 1] = Generator.laplace(df[0][i] + n, df[1][i])
        analytical = analytical_normal_gamma_prior(np.array(measurement[0]), df[1][i])
        numerical = -np.log(integrate.quad(integrand_offset, -30, 30
                                           , args=(df[1][i], np.array(measurement)))[0]) + N * (
                                np.log(2) + np.log(df[1][i]))
        relerr = abs(analytical - numerical) / abs(analytical)
        df[2][i] = relerr
        print(str(i))
    df = pd.DataFrame({'offset': df[0], 'shape': df[1], 'relative error': df[2]})
    pivot = df.pivot(index='offset', columns='shape', values='relative error')
    ax = sns.heatmap(pivot, cmap='coolwarm', ax=ax, linewidths=.5,
                     cbar_kws={'label': 'relative error'}, vmin=0, vmax=1)
    # plt.savefig(fname=d + '\\plots\\relative_error_gaussian_noise.png')
    plt.show()


def main():
    """
    Main
    """
    relative_error()


if __name__ == "__main__":
    main()
