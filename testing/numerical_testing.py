"""
Numerical testing of the analytically derived integral
"""

import scipy.integrate as integrate
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

d = os.getcwd()


def integrand_normal_gamma(offset, precision, measurement):
    """

    :param precision:
    :param offset:
    :param measurement:
    :return:
    """
    alpha = 1
    beta = 1
    kappa = 1
    N = 10
    observable = np.array(range(1, 11))

    factor_0 = precision ** (alpha + (N - 1) / 2)
    factor_1 = np.sum((measurement - observable - offset) ** 2)
    factor_2 = kappa * offset ** 2 + 2 * beta
    result = factor_0 * np.exp(-(precision / 2) * (factor_1 + factor_2))
    return result


def analytical_normal_gamma_prior(measurement):
    """

    :param measurement:
    :return:
    """

    alpha = 1
    beta = 1
    N = 10
    kappa = 1
    observable = np.array(range(1, 11))
    res = measurement - observable
    C_0 = (np.sum(res ** 2) + 2 * beta) / 2
    C_1 = (np.sum(res) ** 2) / (2 * (N + kappa))
    C = C_0 - C_1
    factor_0 = np.sqrt((2 * np.pi) / (N + kappa))
    factor_1 = C ** (-alpha - N / 2)
    return factor_0 * factor_1 * special.gamma(N / 2 + alpha)


def relative_error(noise: str = 'Gaussian', prior: str = 'normal-gamma'):
    """
    Calculating the relative error between our analytical derivation and the numerical integration
    :param noise: Which noise we consider
    :param prior: Which prior was chosen
    """
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    if noise == 'Gaussian':
        Generator = np.random.default_rng()
        df = [np.zeros(21 * 20), np.zeros(21 * 20), np.zeros(21 * 20)]
        for n, value in enumerate(np.linspace(-10, 10, 21)):
            df[0][n * 20:(n + 1) * 20] = value
        precision = [n / 10 for n in range(1, 21)]
        for n, value in enumerate(df[1]):
            df[1][n] = precision[n % 20]
        measurement = [np.ones(10)]
        for i, value in enumerate(df[2]):
            for n in range(1, 11):
                measurement[0][n-1] = Generator.normal(df[0][i] + n, np.sqrt(1 / df[1][i]))
            analytical = analytical_normal_gamma_prior(measurement[0])
            numerical = integrate.dblquad(integrand_normal_gamma, 0, 5, lambda x: -20, lambda x: 20
                                          , args=measurement)[0]
            df[2][i] = abs(analytical - numerical) / analytical
            print(str(i))
        df = pd.DataFrame({'offset': df[0], 'precision': df[1], 'error': df[2]})
        pivot = df.pivot(index='offset', columns='precision', values='error')
        ax = sns.heatmap(pivot, vmin=0.0, vmax=0.1, cmap='coolwarm', ax=ax, linewidths=.5)
        plt.savefig(fname=d + '\\plots\\relative_error_gaussian_noise.png')


def some_checks():
    a = np.array([2, 2])
    b = np.array([3, 3])
    c = (a + b)**2
    e = np.sum(c)
    d = 0


def main():
    """
    Main
    """
    # some_checks()
    relative_error()


if __name__ == "__main__":
    main()
