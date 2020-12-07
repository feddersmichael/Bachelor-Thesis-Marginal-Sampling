import pypesto
import pypesto.petab
import pypesto.sample as sample
import pypesto.visualize as visualize
import seaborn as sns

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

d = os.getcwd()


def negative_log_posterior():
    pass


def negative_log_marginal_posterior():
    pass


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


def merge_full_parameter(save=False):
    data_full_sampling = [0] * 50
    length = 0

    for n in range(50):
        with open(d + '\\Results_CR_FP\\result_FP_CR_' + str(n) + '.pickle', 'rb') as infile_1:
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
    if save:
        with open(d + '\\Results_CR_FP\\merged_data_FP_CR.pickle', 'wb') as save_file:
            pickle.dump(merged_data.sample_result, save_file)

    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    ax0 = visualize.sampling_fval_trace(merged_data, size=(12, 5), full_trace=True)
    ax1 = visualize.sampling_parameters_trace(merged_data, use_problem_bounds=False, full_trace=True, size=(12, 5))
    ax2 = visualize.sampling_1d_marginals(merged_data, size=(12, 5))

    plt.show()


def merge_marginalised_parameter(save=False):
    data_marginal_sampling = [0] * 50
    length = 0

    for n in range(50):
        with open(d + '\\Results_CR_MP\\result_MP_CR_' + str(n) + '.pickle', 'rb') as infile_2:
            data_marginal_sampling[n] = pickle.load(infile_2)
            length += 10001 - data_marginal_sampling[n].burn_in

    merged_data = pypesto.Result(marginal_sampling())

    trace_x = np.zeros((1, length, 2))
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
        burn_in = data_marginal_sampling[n].burn_in
        converge_size = 10001 - burn_in
        merged_data.sample_result.trace_x[0, index:index + converge_size, :] \
            = data_marginal_sampling[n].trace_x[0, burn_in:, :]
        merged_data.sample_result.trace_neglogpost[0, index:index + converge_size] \
            = data_marginal_sampling[n].trace_neglogpost[0, burn_in:]
        merged_data.sample_result.trace_neglogprior[0, index:index + converge_size] \
            = data_marginal_sampling[n].trace_neglogprior[0, burn_in:]
        index += converge_size
    with open(d + '\\Results_CR_MP\\merged_data_MP_CR.pickle', 'wb') as save_file:
        pickle.dump(merged_data.sample_result, save_file)

    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    ax0 = visualize.sampling_fval_trace(merged_data, size=(12, 5), full_trace=True)
    ax1 = visualize.sampling_parameters_trace(merged_data, use_problem_bounds=False, full_trace=True, size=(12, 5))
    ax2 = visualize.sampling_1d_marginals(merged_data, size=(12, 5))

    plt.show()


def one_dimensional_marginal():
    data = [pypesto.Result(standard_sampling()), pypesto.Result(marginal_sampling())]
    with open(d + '\\Results_CR_FP\\merged_data_FP_CR.pickle', 'rb') as data_file:
        data[0].sample_result = pickle.load(data_file)
    with open(d + '\\Results_CR_MP\\merged_data_MP_CR.pickle', 'rb') as data_file:
        data[1].sample_result = pickle.load(data_file)

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(12, 5))

    _, params_fval, _, _, param_names = visualize.sampling.get_data_to_plot(result=data[0], i_chain=0, stepsize=1)

    par_ax = dict(zip(param_names, ax.flat))
    sns.set(style="ticks")

    for idx, par_id in enumerate(param_names):
        sns.distplot(params_fval[par_id], rug=True, ax=par_ax[par_id])

        par_ax[par_id].set_xlabel(param_names[idx])
        par_ax[par_id].set_ylabel('Density')

    sns.despine()

    _, params_fval, _, _, param_names = visualize.sampling.get_data_to_plot(result=data[1], i_chain=0, stepsize=1)
    sns.distplot(params_fval['k1'], rug=True, ax=par_ax['k1'])
    sns.distplot(params_fval['k2'], rug=True, ax=par_ax['k2'])
    fig.tight_layout()
    plt.show()


def boxplot():
    Result_FP = pypesto.Result(standard_sampling())
    Result_MP = pypesto.Result(marginal_sampling())
    x_1 = [0.] * 50
    x_2 = [0.] * 50
    data_sampling = [x_1, x_2]
    for n in range(50):
        with open(d + '\\Results_CR_FP\\result_FP_CR_' + str(n) + '.pickle', 'rb') as infile_1:
            Result_FP.sample_result = pickle.load(infile_1)
            data_sampling[0][n] = pypesto.sample.effective_sample_size(Result_FP)
        with open(d + '\\Results_CR_MP\\result_MP_CR_' + str(n) + '.pickle', 'rb') as infile_2:
            Result_MP.sample_result = pickle.load(infile_2)
            data_sampling[1][n] = pypesto.sample.effective_sample_size(Result_MP)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data_sampling)
    plt.show()


def main():
    # merge_full_parameter(save=True)
    # merge_marginalised_parameter(save=True)
    # one_dimensional_marginal()
    boxplot()
    return 0


main()
