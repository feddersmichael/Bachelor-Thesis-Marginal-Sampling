import scipy.integrate
import math
import numpy as np

rng = np.random.default_rng()


def integrand(c, s, y):  # c as c, s as sigma^2
    prefactor = 1 / ((2 * math.pi * s) ** 5)
    integr = 0
    for k in range(10):
        integr += (y[k] - c - k) ** 2
    integr = math.expm1(integr / (2 * s))
    return prefactor*integr

# for n in range(10)


# ans, err = scipy.integrate.dblquad(integrand, 0, 20, lambda x: -10, lambda x: 10)