import scipy.integrate as integrate
import math
import numpy as np
import scipy.special as special
import scipy
import matplotlib.pyplot as plt


def X_2(t, theta):
    """

    :param t:
    :param theta:
    :return:
    """
    return t


def integrand_flat_prior(c, s, y_bar, h, N):
    """
    Integrand of our marginalisation for flat prior
    :param c:       offset parameter
    :param s:       variance of our noise
    :param y_bar:   the data we can measure offset and noise compensated
    :param h:       the data we can measure, without compensation
    :param N:       amount of time points
    :return:        value of the integrand
    """
    prefactor = (2 * np.pi * s) ** (-N / 2)
    integr = 0
    for k in range(N):  # TODO numpy writing
        integr += (y_bar[k] - c - h[k]) ** 2
    integr = np.exp(integr / (-2 * s))
    return prefactor * integr


def integrand_normal_gamma(c, lamda, y_bar, h, mu, kappa, alpha, beta):
    """

    :param c:       offset parameter
    :param lamda:   = 1/s^2, called precision with sigma being the variance of our noise,
                    also involved in teh variance of c
    :param y_bar:   the data we can measure offset and noise compensated
    :param h:       the data we can measure, without compensation
    :param mu:      mean of distribution of c (normal distribution)
    :param kappa:   rate parameter for the variance of c
    :param alpha:   shape parameter for lambda
    :param beta:    rate parameter for lambda
    :return:        value of the integrand
    """
    N = len(y_bar)
    prefactor = lamda ** (alpha + (N - 1) / 2)
    value = 0
    for k in range(N):
        value += (y_bar[k] - h[k] - c) ** 2
    value += kappa * (c - mu) ** 2 + 2 * beta
    value = np.exp(-value * lamda / 2)
    return value * prefactor


def analytical_flat_prior(y_bar, h, N):
    """
    analytical solution for our integral, marginalizing sigma and c
    :param y_bar:   the data we can measure offset and noise compensated
    :param h:       the data we can measure, without compensation
    :param N:       amount of time stamps
    :return:        value of the integral
    """
    product_1 = N ** (N / 2 - 2)
    product_2 = 2 * np.pi ** ((N - 1) / 2)
    sum_1 = N * np.sum((h - y_bar) ** 2)
    sum_2 = np.sum(y_bar - h) ** 2
    product_3 = (sum_1 - sum_2) ** ((N - 3) / -2)
    product_4 = special.gamma((N - 3) / 2)
    return product_1 * product_3 * product_4 / product_2


def analytical_normal_gamma_prior(y_bar, h, mu, kappa, alpha, beta):
    """

    :param y_bar:
    :param h:
    :param mu:
    :param kappa:
    :param alpha:
    :param beta:
    :return:
    """
    N = len(y_bar)
    C_1 = 0
    C_2 = 0
    for k in range(N):
        temp = y_bar[k] - h[k]
        C_1 += temp ** 2
        C_2 += temp
    C_1 += kappa * mu ** 2 + 2 * beta
    C_2 = (C_2 + kappa * mu) ** 2
    C = C_1 / 2 - C_2 / (2 * (N + kappa))
    prefactor_1 = (beta/C) ** alpha * np.sqrt(kappa / (N + kappa))
    prefactor_2 = scipy.special.gamma(alpha) * (2 * np.pi * C) ** (N / 2)
    return prefactor_1 / prefactor_2 * scipy.special.gamma(N / 2 + alpha)


def relative_error_srm(prior, N=10, size=20):
    """

    :param prior:
    :param N:
    :param size:
    :return:
    """
    Y, X = np.meshgrid(np.linspace(-10, 10 + (20 / size), size + 1), np.linspace(0.05, 2 + (1.95 / size), size + 1))
    Z = np.empty((size, size))
    h = np.array([i + 1 for i in range(N)])
    y_bar = np.empty(N)
    if prior == 'flat':
        for sigma in range(size):
            for c in range(size):
                for n in range(N):
                    y_bar[n] = np.random.normal(Y[c, sigma] + h[n], X[c, sigma])
                mean_1 = integrate.dblquad(integrand_flat_prior, 0, 5 * X[c, sigma], lambda x: Y[c, sigma] - 5,
                                           lambda x: Y[c, sigma] + 5, args=(y_bar, h, N))[0]
                mean_2 = analytical_flat_prior(y_bar, h, N)
                Z[c, sigma] = abs(mean_1 - mean_2) / mean_2
    if prior == 'normal_gamma':
        mu = 0
        kappa = 1
        alpha = 1
        beta = 1
        Int_const = beta ** alpha * np.sqrt(kappa)
        Int_const = Int_const / (scipy.special.gamma(alpha) * (2 * np.pi) ** ((N + 1) / 2))
        for sigma in range(size):
            for c in range(size):
                for n in range(N):
                    y_bar[n] = np.random.normal(Y[c, sigma] + h[n], X[c, sigma])
                mean_1 = integrate.dblquad(integrand_normal_gamma, 0, 10, lambda x: Y[c, sigma] - 5,
                                           lambda x: Y[c, sigma] + 5, args=(y_bar, h, mu, kappa, alpha, beta))[0] \
                         * Int_const
                mean_2 = analytical_normal_gamma_prior(y_bar, h, mu, kappa, alpha, beta)
                Z[c, sigma] = abs(mean_1 - mean_2) / (mean_2 + 1e-12)
    out = [X, Y, Z]
    return out


def relative_error_crm(prior, N=10, size=20):
    """

    :param prior:
    :param N:
    :param size:
    :return:
    """
    X = np.linspace(-10, 10, size)
    Y = np.empty(size)
    h = np.empty(N)
    y_bar = np.empty(N)
    theta = np.array((1, 1))
    sigma = 0.2
    if prior == 'flat':
        for c in range(size):
            for n in range(N):
                h[n] = X_2(n + 1, theta)
                y_bar[n] = np.random.normal(X[c] + h[n], sigma)
            mean_1 = integrate.dblquad(integrand_flat_prior, 0, 2, lambda x: X[c] - 5,
                                       lambda x: X[c] + 5, args=(y_bar, h, N))[0]
            mean_2 = analytical_flat_prior(y_bar, h, N)
            Y[c] = abs(mean_1 - mean_2) / mean_2
    if prior == 'normal_gamma':
        mu = 0
        kappa = 1
        alpha = 1
        beta = 1
        Int_const = beta ** alpha * np.sqrt(kappa)
        Int_const = Int_const / (scipy.special.gamma(alpha) * (2 * np.pi) ** ((N + 1) / 2))
        for c in range(size):
            for n in range(N):
                h[n] = X_2(n + 1, theta)
                y_bar[n] = np.random.normal(X[c] + h[n], sigma)
            mean_1 = integrate.dblquad(integrand_normal_gamma, 0, 2, lambda x: X[c] - 5,
                                       lambda x: X[c] + 5, args=(y_bar, h, mu, kappa, alpha, beta))[0] \
                     * Int_const
            mean_2 = analytical_normal_gamma_prior(y_bar, h, mu, kappa, alpha, beta)
            Y[c] = abs(mean_1 - mean_2) / mean_2
    out = [X, Y]
    return out


def numerical_testing(model='srm', prior='flat'):
    """
    General environment for numerical testing. We evaluate the relative error for several values of sigma and c.
    The error is plotted against the changing values of c and sigma.
    :param prior: flat or normal gamma
    :param model: determines model - either sample regression or conversion reaction
    :return: plot of the relative error
    """
    if model == 'crm':
        if prior == 'flat':
            return relative_error_crm('flat')
        else:
            return relative_error_crm('normal_gamma')
    else:
        if prior == 'flat':
            return relative_error_srm('flat')
        else:
            return relative_error_srm('normal_gamma')


def main():
    # result_1 = numerical_testing('srm', 'flat')
    result_2 = numerical_testing('srm', 'normal_gamma')
    k = 1


if __name__ == "__main__":
    main()
