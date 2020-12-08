import pypesto
import pypesto.petab
import pypesto.sample as sample
import pypesto.visualize as visualize
import seaborn as sns
from copy import deepcopy

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd

d = os.getcwd()


def negative_log_posterior():
    pass


def negative_log_marginal_posterior():
    pass


def standard_sampling_CR():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5, -np.inf, 0],  # lower bounds
                              ub=[5, 5, np.inf, np.inf],  # upper bounds
                              x_names=['k1', 'k2', 'offset', 'precision'],  # parameter names
                              x_scales=['log', 'log', 'lin', 'lin'])  # parameter scale
    return problem


def marginal_sampling_CR():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_marginal_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5],  # lower bounds
                              ub=[5, 5],  # upper bounds
                              x_names=['k1', 'k2'],  # parameter names
                              x_scales=['log', 'log'])  # parameter scale
    return problem


def standard_sampling_mRNA():
    """Creates a pyPESTO problem."""
    df = pd.read_csv('mRNA-transfection/data.csv', sep='\t')
    objective = pypesto.Objective(fun=negative_log_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-2, -5, -5, -5, -np.inf, 0],  # lower bounds
                              ub=[np.log10(df.Time.max()), 5, 5, 5, np.inf, np.inf],  # upper bounds
                              x_names=['t_0', 'k_{TL}*m_0', 'xi', 'delta',
                                       'offset', 'precision'],  # parameter names
                              x_scales=['log10', 'log10', 'log10', 'log10',
                                        'lin', 'lin'])  # parameter scale
    return problem


def marginal_sampling_mRNA():
    """Creates a pyPESTO problem."""
    df = pd.read_csv('mRNA-transfection/data.csv', sep='\t')
    objective = pypesto.Objective(fun=negative_log_marginal_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-2, -5, -5, -5],  # lower bounds
                              ub=[np.log10(df.Time.max()), 5, 5, 5],  # upper bounds
                              x_names=['t_0', 'k_{TL}*m_0', 'xi', 'delta'],  # parameter names
                              x_scales=['log10', 'log10', 'log10', 'log10'])  # parameter scale
    return problem


def merge_and_plot(model: str = 'CR', sampling_type: str = 'FP', save: bool = False, visualization: bool = True):
    data_full_sampling = [0] * 50
    length = 0
    if model == 'CR':
        amount_samples = 50
        states = 10001
        if sampling_type == 'FP':
            problem = standard_sampling_CR()
            parameters = 4
            storage_ID = '_CR_FP'
        if sampling_type == 'MP':
            problem = marginal_sampling_CR()
            parameters = 2
            storage_ID = '_CR_MP'
    elif model == 'mRNA':
        amount_samples = 10
        states = 1000001
        if sampling_type == 'FP':
            problem = standard_sampling_mRNA()
            parameters = 6
            storage_ID = '_mRNA_FP'
        if sampling_type == 'MP':
            problem = marginal_sampling_mRNA()
            parameters = 4
            storage_ID = '_mRNA_MP'

    for n in range(amount_samples):
        with open(d + '\\Results' + storage_ID + '\\result' + storage_ID + '_' + str(n) + '.pickle', 'rb') as infile_1:
            data_full_sampling[n] = pickle.load(infile_1)  # pickle.load(infile_1)
            length += states - data_full_sampling[n].burn_in

    merged_data = pypesto.Result(problem)

    trace_x = np.zeros((1, length, parameters))
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
    for n in range(amount_samples):
        burn_in = data_full_sampling[n].burn_in
        converge_size = states - burn_in
        merged_data.sample_result.trace_x[0, index:index + converge_size, :] \
            = data_full_sampling[n].trace_x[0, burn_in:, :]
        merged_data.sample_result.trace_neglogpost[0, index:index + converge_size] \
            = data_full_sampling[n].trace_neglogpost[0, burn_in:]
        merged_data.sample_result.trace_neglogprior[0, index:index + converge_size] \
            = data_full_sampling[n].trace_neglogprior[0, burn_in:]
        index += converge_size
    if save:
        with open(d + '\\Results' + storage_ID + '\\merged_data' + storage_ID + '.pickle', 'wb') as save_file:
            pickle.dump(merged_data.sample_result, save_file)

    if visualization:
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)

        ax0 = visualize.sampling_fval_trace(merged_data, size=(12, 5), full_trace=True)
        ax1 = visualize.sampling_parameters_trace(merged_data, use_problem_bounds=False, full_trace=True, size=(12, 5))
        ax2 = visualize.sampling_1d_marginals(merged_data, size=(12, 5))

        plt.show()


def one_dimensional_marginal(model: str = 'CR'):
    if model == 'CR':
        problem_1 = standard_sampling_CR()
        problem_2 = marginal_sampling_CR()
        storage_ID = '_CR'

    elif model == 'mRNA':
        problem_1 = standard_sampling_mRNA()
        problem_2 = marginal_sampling_mRNA()
        storage_ID = '_mRNA'

    data = [pypesto.Result(problem_1), pypesto.Result(problem_2)]
    with open(d + '\\Results' + storage_ID + '_FP\\merged_data' + storage_ID + '_FP.pickle', 'rb') as data_file:
        data[0].sample_result = pickle.load(data_file)
    with open(d + '\\Results' + storage_ID + '_MP\\merged_data' + storage_ID + '_MP.pickle', 'rb') as data_file:
        data[1].sample_result = pickle.load(data_file)

    nr_params, params_fval, _, _, param_names = visualize.sampling.get_data_to_plot(result=data[0], i_chain=0, stepsize=1)

    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    fig, ax = plt.subplots(num_row, num_col, squeeze=False, figsize=(12, 5))

    par_ax = dict(zip(param_names, ax.flat))
    sns.set(style="ticks")

    for idx, par_id in enumerate(param_names):
        sns.distplot(params_fval[par_id], rug=True, ax=par_ax[par_id])

        par_ax[par_id].set_xlabel(param_names[idx])
        par_ax[par_id].set_ylabel('Density')

    _, params_fval, _, _, param_names = visualize.sampling.get_data_to_plot(result=data[1], i_chain=0, stepsize=1)

    for n in param_names:
        sns.distplot(params_fval[n], rug=True, ax=par_ax[n])

    sns.despine()
    fig.tight_layout()
    plt.show()


def boxplot(mode: str = 'CPU'):
    Result_FP = pypesto.Result(standard_sampling_CR())
    Result_MP = pypesto.Result(marginal_sampling_CR())
    x_1 = [0.] * 50
    x_2 = [0.] * 50
    eff_sample_size_per_CPU = [x_1, x_2]
    CPU_time = deepcopy(eff_sample_size_per_CPU)
    for n in range(50):
        with open(d + '\\Results_CR_FP\\result_CR_FP_' + str(n) + '.pickle', 'rb') as infile_1:
            Result_FP.sample_result = pickle.load(infile_1)
            eff_sample_size_per_CPU[0][n] = pypesto.sample.effective_sample_size(Result_FP) \
                                            / Result_FP.sample_result.time
            CPU_time[0][n] = Result_FP.sample_result.time
        with open(d + '\\Results_CR_MP\\result_CR_MP_' + str(n) + '.pickle', 'rb') as infile_2:
            Result_MP.sample_result = pickle.load(infile_2)
            eff_sample_size_per_CPU[1][n] = pypesto.sample.effective_sample_size(
                Result_MP) / Result_MP.sample_result.time
            CPU_time[1][n] = Result_MP.sample_result.time

    if mode == 'CPU':
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot()
        ax.boxplot(CPU_time, labels=['Full parameter', 'Marginal parameter'])
        ax.set_ylabel('CPU-time')
        plt.show()
    elif mode == 'eff_ss_CPU':
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot()
        ax.boxplot(eff_sample_size_per_CPU, labels=['Full parameter', 'Marginal parameter'])
        ax.set_ylabel('Effective sample size per CPU time')
        plt.show()


def main():
    # merge_and_plot('CR', 'FP', True, True)
    # merge_marginalised_parameter(save=True)
    # one_dimensional_marginal('CR')
    boxplot('CPU')
    return 0


main()
