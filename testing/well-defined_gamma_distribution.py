import numpy as np

N = 1
# h = np.array([1, 1, 1, 1, 1])
h = np.array([1])
# y = np.array([1, 1, 1, 1, 1])
y = np.array([1])
kappa = 1
mu = 1
tau = 2
z = 1

sum_1 = np.sum(h ** 2) * (N * kappa * mu ** 2 + (N + kappa) * tau * z ** 2 + (N + kappa) * np.sum(y ** 2)
                          - 2 * kappa * mu * np.sum(y) - np.sum(y) ** 2)

sum_2 = np.sum(h) * (2 * (kappa * mu + np.sum(y)) * (tau * z + np.sum(h * y))
                     - np.sum(h) * (kappa * mu ** 2 + tau * z ** 2 + np.sum(y ** 2)))

sum_3 = tau * (N * kappa * mu ** 2 + (N + kappa) * np.sum(y ** 2) - 2 * kappa * mu * np.sum(y)
               - np.sum(y) ** 2 - (N + kappa) * (2 * tau * z * np.sum(h * y)))

sum_4 = -(N + kappa) * np.sum(h * y) ** 2

# print(sum_1)

# print(sum_2)

# print(sum_3)

# print(sum_4)

print(sum_4 + sum_3 + sum_2 + sum_1)

C_1 = kappa * mu ** 2 + tau * z ** 2 + np.sum(y ** 2) - (kappa * mu + np.sum(y)) ** 2 / (N + kappa)

C_2 = ((kappa * mu + np.sum(y)) * (np.sum(h)) - (N + kappa) * (tau * z + np.sum(h * y))) ** 2

C_3 = (N + kappa) * ((N + kappa) * (tau + np.sum(h ** 2)) - (np.sum(h)) ** 2)

C = C_1 - C_2 / C_3

print(C)

t_1 = tau * (N * kappa * mu ** 2 + (N + kappa) * np.sum(y ** 2)) + np.sum(h ** 2) * (
        N * kappa * mu ** 2 + (N + kappa) * tau * z ** 2 + (N + kappa) * np.sum(y ** 2))

t_2 = 2 * (kappa * mu + np.sum(y)) * np.sum(h) * (tau * z + np.sum(h * y))

t_3 = (tau + np.sum(h ** 2)) * (2 * kappa * mu * np.sum(y) + np.sum(y) ** 2) + np.sum(h) ** 2 * (
        kappa * mu ** 2 + tau * z ** 2 + np.sum(y ** 2))

t_4 = (N + kappa) * (2 * tau * z * np.sum(h * y) + np.sum(h * y) ** 2)

t = t_1 + t_2 - t_3 - t_4

print(t)
