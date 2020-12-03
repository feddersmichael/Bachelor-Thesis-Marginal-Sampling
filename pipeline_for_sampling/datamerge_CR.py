import pypesto
import pypesto.petab
import pypesto.sample as sample
import pypesto.visualize as visualize

import petab
import numpy as np
from scipy.special import gammaln
import pickle
import os
import matplotlib.pyplot as plt

petab_problem = petab.Problem.from_yaml(
    "conversion_reaction/SS_conversion_reaction.yaml")

d = os.getcwd()
mu = 0
alpha = 100
beta = 0.1
kappa = 0.01


def analytical_b(t, a0, b0, k1, k2):
    return (k2 - k2 * np.exp(-(k2 + k1) * t)) / (k2 + k1)


def simulate_model(x, tvec):
    # assign parameters
    k1, k2, _, _ = x
    # define initial conditions
    a0 = 1
    b0 = 0.01
    # simulate model
    simulation = [analytical_b(t, a0, b0, k1, k2)
                  for t in tvec]
    return simulation


def log_prior(x):
    """ Log prior function."""
    # assign variables from input x
    offset = x[2]
    precision = x[3]

    # evaluate log normal-gamma prior
    l_prior = alpha * np.log(beta) + (alpha - 0.5) * np.log(precision) \
              + 0.5 * (np.log(kappa) - np.log(2) - np.log(np.pi)) - gammaln(alpha) \
              - beta * precision - 0.5 * precision * kappa * (offset - mu) ** 2

    return l_prior


def negative_log_posterior(x):
    """ Negative log posterior function."""

    offset = x[2]
    precision = x[3]

    # experimental data
    data = np.asarray(petab_problem.measurement_df.measurement)
    # time vector
    tvec = np.asarray(petab_problem.measurement_df.time)

    n_timepoints = len(tvec)

    # simulate model
    _simulation = simulate_model(np.exp(x), tvec)
    simulation = (offset + np.asarray(_simulation))

    # evaluate standard log likelihood
    res = data - simulation
    sum_res = np.sum(res ** 2)

    constant = np.log(precision) - np.log(2) - np.log(np.pi)

    l_llh = 0.5 * (n_timepoints * constant - precision * sum_res)

    # evaluate log normal-gamma prior
    l_prior = log_prior(x)

    # return NEGATIVE log posterior (required for pyPESTO)
    return -(l_llh + l_prior)


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


def standard_sampling():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5, -np.inf, 0],  # lower bounds
                              ub=[5, 5, np.inf, np.inf],  # upper bounds
                              x_names=['k1', 'k2', 'offset', 'precision'],  # parameter names
                              x_scales=['log', 'log', 'lin', 'lin'])  # parameter scale
    return problem


def marginal_sampling():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_marginal_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5],  # lower bounds
                              ub=[5, 5],  # upper bounds
                              x_names=['k1', 'k2'],  # parameter names
                              x_scales=['log', 'log'])  # parameter scale
    return problem


def merge_full_parameter():
    data_full_sampling = [0] * 50
    length = 0

    for n in range(50):
        with open(d + '\\Results_FP_CR\\result_FP_CR_' + str(n) + '.pickle', 'rb') as infile_1:
            data_full_sampling[n] = pickle.load(infile_1)  # pickle.load(infile_1)
            length += 10001 - data_full_sampling[n].burn_in

    merged_data = pypesto.Result(standard_sampling())

    trace_x = np.zeros((1, length, 4))
    merged_data.sample_result.trace_x = trace_x

    trace_neglogpost = np.zeros((1, length))
    merged_data.sample_result.trace_neglogpost = trace_neglogpost

    trace_neglogprior = np.zeros((1, length))
    merged_data.sample_result.trace_neglogprior = trace_neglogprior

    merged_data.sample_result.betas = np.array([1])

    merged_data.sample_result.burn_in = 0

    merged_data.sample_result.auto_correlation = None
    sample.effective_sample_size(merged_data)

    index = 0
    for n in range(50):
        burn_in = data_full_sampling[n].burn_in
        converge_size = 10001 - burn_in
        merged_data.sample_result.trace_x[0, index:index + converge_size, :] \
            = data_full_sampling[n].trace_x[0, burn_in:, :]
        merged_data.sample_result.trace_neglogpost[0, index:index + converge_size] \
            = data_full_sampling[n].trace_neglogpost[0, burn_in:]
        merged_data.sample_result.trace_neglogprior[0, index:index + converge_size] \
            = data_full_sampling[n].trace_neglogprior[0, burn_in:]
        index += converge_size
    with open(d + '\\Results_FP_CR\\merged_data_FP_CR.pickle', 'wb') as save_file:
        pickle.dump(merged_data, save_file)

    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    ax0 = visualize.sampling_fval_trace(merged_data, size=(12, 5), full_trace=True)
    ax1 = visualize.sampling_parameters_trace(merged_data, use_problem_bounds=False, full_trace=True, size=(12, 5))
    ax2 = visualize.sampling_1d_marginals(merged_data, size=(12, 5))

    plt.show()


def merge_marginalised_parameter():
    data_marginal_sampling = [0] * 50
    for n in range(50):
        with open(d + '\\Results_MP_CR\\result_MP_CR_' + str(n) + '.pickle', 'rb') as infile_2:
            data_marginal_sampling[n] = pickle.load(infile_2)


def main():
    merge_full_parameter()
    n = 1
    return 0


main()
