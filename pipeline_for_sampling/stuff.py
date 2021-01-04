import pypesto
import pypesto.petab
import seaborn as sns
import matplotlib.pyplot as plt

import petab
import numpy as np
from scipy.special import gammaln
import pickle
import os

# path of the directory
d = os.getcwd()
# import to petab
petab_problem = petab.Problem.from_yaml(
    "conversion_reaction/SS_conversion_reaction.yaml")

def analytical_b(t, a0, b0, k1, k2):
    return (k2 - k2 * np.exp(-(k2 + k1) * t)) / (k2 + k1)

def simulate_model(x, tvec):
    # assign parameters
    k1, k2 = x
    # define initial conditions
    a0 = 1
    b0 = 0.01
    # simulate model
    simulation = [analytical_b(t, a0, b0, k1, k2)
                   for t in tvec]
    return simulation

mu=0
# std for scaling parameter --> higher = more constrained / lower = more relaxed
alpha=100
# center the sigma parameter
beta=0.1
# std for scaling parameter --> higher = more constrained / lower = more relaxed
kappa=0.01

def negative_log_marginal_posterior(x):
    """
    negative logarithmic marginalized posterior
    :param x: x_0 = k1, x_1 = k_2
    """

    # experimental data
    data = np.asarray(petab_problem.measurement_df.measurement)
    # time vector
    tvec = np.asarray(petab_problem.measurement_df.time)

    n_timepoints = len(tvec)

    # simulate model
    _simulation = simulate_model(np.exp(x), tvec)
    simulation = np.asarray(_simulation)

    # evaluate standard log likelihood
    res = data - simulation

    C_1 = (np.sum(res ** 2) + kappa * mu ** 2 + 2 * beta) / 2
    C_2 = ((np.sum(res) + kappa * mu) ** 2) / (2 * (n_timepoints + kappa))
    log_C = np.log(C_1 - C_2)

    mlikelihood_1 = alpha * ((np.log(beta)) - log_C)
    mlikelihood_2 = gammaln(alpha)
    mlikelihood_3 = (n_timepoints / 2) * (np.log(2) + np.log(np.pi) + log_C)
    mlikelihood_4 = (np.log(kappa) - np.log(n_timepoints + kappa)) / 2
    mlikelihood_5 = gammaln(n_timepoints / 2 + alpha)
    marg_likelihood = mlikelihood_1 - mlikelihood_2 - mlikelihood_3 + mlikelihood_4 + mlikelihood_5

    return -marg_likelihood

def marginal_sampling():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_marginal_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5],  # lower bounds
                              ub=[5, 5],  # upper bounds
                              x_names=['k1', 'k2'],  # parameter names
                              x_scales=['log', 'log'])  # parameter scale
    return problem

tvec = np.asarray(petab_problem.measurement_df.time)
N = len(tvec)
data_model = np.asarray(petab_problem.measurement_df.measurement)
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))


def Constant(x):
    _simulation = simulate_model(np.exp(x), tvec)
    simulation = np.asarray(_simulation)

    res = data_model - simulation

    summand_1 = (np.sum(res ** 2) + kappa * mu ** 2 + 2 * beta) / 2
    summand_2 = (1 / (2 * (N + kappa))) * (np.sum(res) + kappa * mu) ** 2

    return summand_1 - summand_2


Generator = np.random.default_rng()
results = pypesto.Result(marginal_sampling())

with open(d + '\\Results_CR_MP\\merged_data_CR_MP.pickle', 'rb') as infile:
    results.sample_result = pickle.load(infile)[0]

precision_list = np.zeros(np.shape(results.sample_result.trace_x)[1])

for index, data in enumerate(results.sample_result.trace_x[0, :, :]):
    shape = alpha + N / 2
    scale = 1 / Constant(data)
    precision_list[index] = Generator.gamma(shape, scale)

sns.distplot(precision_list, rug=True, axlabel='precision', ax=axs[1])


def mu_(x):
    _simulation = simulate_model(np.exp(x), tvec)
    simulation = np.asarray(_simulation)

    res = data_model - simulation
    result_ = np.sum(res) + kappa * mu
    return result_ / (N + kappa)


offset_list = np.zeros(np.shape(results.sample_result.trace_x)[1])

for index, data in enumerate(results.sample_result.trace_x[0, :, :]):
    new_mu = mu_(data)
    new_sigmasquare = 1 / ((N + kappa) * precision_list[index])
    offset_list[index] = Generator.normal(new_mu, new_sigmasquare)

sns.distplot(offset_list, rug=True, axlabel='offset', ax=axs[0])

sns.despine(fig)
plt.tight_layout()
plt.savefig(fname=d + '\\plots\\CR_MP\\offset_and_precision.png')

with open('Results_CR_MP\\offset_and_precision.pickle', 'wb')as file:
    pickle.dump([offset_list, precision_list], file)
