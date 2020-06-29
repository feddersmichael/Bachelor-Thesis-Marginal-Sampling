import scipy.integrate as integrate
import math
import numpy as np
import scipy.special as special
import scipy
import matplotlib.pyplot as plt


def h_func_srm(t, theta):
    """

    :param t:
    :param theta:
    :return:
    """
    """
    observation function
    """
    return t


def X_2(t, theta):
    """

    :param t:
    :param theta:
    :return:
    """
    return t


def h_func_crm(t, theta):
    """

    :param t:
    :param theta:
    :return:
    """
    """
    observation function
    """
    return X_2(t, theta)


def y_func_offset(c, h_t_theta):
    """

    :param c:
    :param h_t_theta:
    :return:
    """
    """
    observation function with a transformation factor
    """
    return c + h_t_theta


def y_meas_add_gaus(c, h_t_theta, sigma):
    """

    :param c:
    :param h_t_theta:
    :param sigma:
    :return:
    """
    """
    Adds noise to data y.

    We get the data y as an input and simulate
    random noise which represents the error.
    The measured data y_tilde gets returned.
    """
    return np.random.normal(y_func_offset(c, h_t_theta), sigma)


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


def integrand_normal_gamma(c, lamda, y_bar, h, mu, kappa, alpha, beta, N):
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
    :param N:       amount of time points
    :return:        value of the integrand
    """
    prefactor_1 = beta ** alpha * np.sqrt(kappa) * np.e ** (2 * beta)
    prefactor_2 = scipy.special.gamma(alpha) * (2 * np.pi) ** ((N + 1) / 2)
    prefactor_3 = lamda ** ((N + 2 * alpha - 1) / 2)
    value = 0
    for k in range(N):
        value += (y_bar[k] - h[k] - c) ** 2
    value += kappa * (c - mu) ** 2
    value = np.exp(-value * lamda / 2)
    return value * (prefactor_1 / prefactor_2) * prefactor_3


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


def analytical_normal_gamma_prior(y_bar, h, mu, kappa, alpha, beta, N):
    """

    :param y_bar:
    :param h:
    :param mu:
    :param kappa:
    :param alpha:
    :param beta:
    :param N:
    :return:
    """
    C_1 = 0
    C_2 = 0
    for k in range(N):
        temp = y_bar[k] - h[k]
        C_1 += temp ** 2
        C_2 += temp
    C_1 += kappa * mu ** 2
    C_2 = (C_2 + kappa * mu) ** 2
    C = C_1 / 2 - C_2 / (2 * (N + kappa))
    prefactor_1 = beta ** alpha * np.sqrt(kappa) * np.e ** (2 * beta)
    prefactor_2 = scipy.special.gamma(alpha) * (2 * np.pi) ** (N / 2) * (N + kappa) * C ** ((N + 2 * alpha) / 2)
    return prefactor_1 / prefactor_2 * scipy.special.gamma((N + 2 * alpha) / 2)


def relative_error_srm(c, sigma, theta, N):
    h = np.zeros(N)
    y = np.zeros(N)
    for n in range(N):
        h[n] = h_func_srm(n + 1, theta)
        y[n] = y_meas_add_gaus(c, h[n], sigma)
    mean_1 = integrate.dblquad(integrand_flat_prior, 0, min(10, 5 * sigma), lambda x: -5 + c, lambda x: 5 + c,
                               args=(y, h, N))[0]
    mean_2 = analytical_flat_prior(y, h, N)
    return abs(mean_1 - mean_2) / mean_2


def relative_error_crm(c, sigma, theta, N):
    h = np.zeros(N)
    y_bar = np.zeros(N)
    mu = 0
    kappa = 1
    alpha = 1
    beta = 1
    for n in range(N):
        h[n] = h_func_crm(n + 1, theta)
        y_bar[n] = y_meas_add_gaus(c, h[n], sigma)
    mean_1 = integrate.dblquad(integrand_normal_gamma, 0, min(10, 5 * sigma), lambda x: -5 + c, lambda x: 5 + c,
                               args=(y_bar, h, mu, kappa, alpha, beta, N))[0]
    mean_2 = analytical_normal_gamma_prior(y_bar, h, mu, kappa, alpha, beta, N)
    return abs(mean_1 - mean_2) / mean_2


def numerical_testing(mode=0):
    """
    General environment for numerical testing. We evaluate the relative error for several values of sigma and c.
    :param mode: mode decides the type
    mode 0 is sample regression model
    mode 1 is conversion reaction model
    :return: plot of the relative error
    """

    c = np.linspace(-10, 10, 20)
    N = 10
    h = np.zeros(N)
    y = np.zeros(N)
    if mode == 1:
        theta = np.array((1, 1))
        sigma = 0.2
    else:  # mode == 0
        theta = 0
        sigma = np.linspace(0.02, 2, 20)  # sigma in the sense of sigma^2
    for n in range(N):
        if mode == 1:
            h[n] = h_func_crm(n + 1, theta)
        else:  # mode == 0
            h[n] = h_func_srm(n + 1, theta)
        y[n] = y_meas_add_gaus(c, h[n], sigma)
    if mode == 1:
        value = np.empty(c.size)
        for i in range(c.size):
            value[i] = relative_error_crm(c[i], sigma, theta, N)
    else:  # mode == 0
        value = np.empty((c.size, sigma.size))
        for i in range(c.size):
            for j in range(sigma.size):
                value[i, j] = relative_error_srm(c[i], sigma[j], theta, N)
        fig = plt.figure()
        fig, ax = fig.add_subplot(1, 1, 1, xlabel='$\\sigma$', ylabel='c')
        im = plt.imshow(value, origin='lower', interpolation='bilinear',
                        aspect='equal')  # extent=(sigma[0], sigma[-1], c[0], c[-1])
        fig.colorbar(im)
        fig.savefig('heatmap.png')
        plt.show()

    # solution = analytical(y, h, N)
    # result = integrate.dblquad(integrand, 0, 1000, lambda x: -5+c, lambda x: 5+c, args=(y, h, N))


def main():
    numerical_testing()


if __name__ == "__main__":
    main()
