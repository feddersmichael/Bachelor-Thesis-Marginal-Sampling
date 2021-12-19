import numpy as np
import scipy.special
from scipy import integrate

y = np.array([1.0219, 1.4442, 1.584, 1.6951, 1.7249, 1.7242, 1.7398, 1.7506, 1.7843, 1.7768, 1.7359])
N = 11

beta = 2
alpha = 3
lamda = 0.8
kappa = 4
mu = -2
z = 2
tau = 3
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def analytical_b(t, a0, b0, k1, k2):
    return (k2 - k2 * np.exp(-(k2 + k1) * t)) / (k2 + k1)


h = np.array([analytical_b(j, 1, 0, 1, 1) for j in t])

f = lambda c, s: np.exp(- (lamda / 2) * (np.sum((y - c - s * h) ** 2) + kappa * (c - mu) ** 2 + tau * (s - z) ** 2
                                         + 2 * beta)) * lamda ** (N / 2 + alpha) * beta ** alpha \
                 * np.sqrt(kappa * tau) * (1 / scipy.special.gamma(alpha)) * (2 * np.pi) ** (-(N + 2) / 2)

numerical_sol_integr_c_s = integrate.dblquad(f, np.NINF, np.inf, np.NINF, np.inf)

konst1 = (beta ** alpha / (scipy.special.gamma(alpha) * (2 * np.pi) ** (N / 2))) * np.sqrt(kappa * tau)
konst2 = 1 / np.sqrt((N + kappa) * (tau + np.sum(h ** 2)) - np.sum(h) ** 2)
konst3 = lamda ** (alpha + (N / 2) - 1) * np.exp(-(lamda / 2) * (kappa * mu ** 2 + tau * z ** 2 + 2 * beta
                                                                 + np.sum(y ** 2) - (
                                                                         (kappa * mu + np.sum(y)) ** 2 / (N + kappa))))
konst4 = np.exp((lamda / 2) * (((kappa * mu + np.sum(y)) * np.sum(h) - (N + kappa) * (tau * z + np.sum(h * y))) ** 2
                               / ((N + kappa) * ((N + kappa) * (tau + np.sum(h ** 2)) - np.sum(h) ** 2))))

analytical_sol_integr_c_s = konst1 * konst2 * konst3 * konst4

const1 = (kappa * mu ** 2 + tau * z ** 2 + 2 * beta + np.sum(y ** 2) - ((kappa * mu + np.sum(y)) ** 2) / (
    N + kappa)) / 2
const2 = ((kappa * mu + np.sum(y)) * np.sum(h) - (N + kappa) * (tau * z + np.sum(h * y))) ** 2
const3 = 2 * (N + kappa) * ((N + kappa) * (tau + np.sum(h ** 2)) - np.sum(h) ** 2)
C = const1 - const2 / const3

analytical_sol_integr_lamda = ((beta / C) ** alpha / (
        scipy.special.gamma(alpha) * (C * 2 * np.pi) ** (N / 2))) * np.sqrt(kappa * tau) \
                              * (1 / np.sqrt(
    (N + kappa) * (tau + np.sum(h ** 2)) - np.sum(h) ** 2)) * scipy.special.gamma(alpha + N / 2)

g = lambda lamda: lamda ** (alpha + (N / 2) - 1) * np.exp(-(lamda / 2) * (kappa * mu ** 2 + tau * z ** 2 + 2 * beta
                                                                          + np.sum(y ** 2) - (
                                                                                  (kappa * mu + np.sum(y)) ** 2 / (
                                                                                      N + kappa))) + (lamda / 2)
                                                          * (((kappa * mu + np.sum(y)) * np.sum(h) - (N + kappa) * (
        tau * z + np.sum(h * y))) ** 2
                                                             / ((N + kappa) * (
            (N + kappa) * (tau + np.sum(h ** 2)) - np.sum(h) ** 2))))

numerical_sol_integr_lamda = konst1 * konst2 * integrate.quad(g, 0, np.inf)[0]

print(analytical_sol_integr_c_s)
print(numerical_sol_integr_c_s)

print(analytical_sol_integr_lamda)
print(analytical_sol_integr_lamda)
