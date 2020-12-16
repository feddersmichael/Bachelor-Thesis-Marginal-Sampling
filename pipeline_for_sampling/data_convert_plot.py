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
import matplotlib
import pandas as pd

d = os.getcwd()


def negative_log_posterior():
    """
    dummy function for our problem function
    """
    pass


def negative_log_marginal_posterior():
    """
    dummy function for our problem function
    """
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


def result_generator(result_type: str = None, sample_result: str = None):
    """
    Transforms a sample_result into a proper Result object
    :param result_type: Defines the object function we use
    :param sample_result: if given aleady includes the sample_result into the new Result object
    :return: the generated Result object
    """
    if result_type == 'CR_FP':
        generated_result = pypesto.Result(standard_sampling_CR())
        if sample_result is not None:
            generated_result.sample_result = sample_result
        return generated_result
    elif result_type == 'CR_MP':
        generated_result = pypesto.Result(marginal_sampling_CR())
        if sample_result is not None:
            generated_result.sample_result = sample_result
        return generated_result
    elif result_type == 'mRNA_FP':
        generated_result = pypesto.Result(standard_sampling_mRNA())
        if sample_result is not None:
            generated_result.sample_result = sample_result
        return generated_result
    elif result_type == 'mRNA_MP':
        generated_result = pypesto.Result(marginal_sampling_mRNA())
        if sample_result is not None:
            generated_result.sample_result = sample_result
        return generated_result


def burn_in_change(path: str = None, burn_in: int = None):
    """
    Changes the burn_in parameter manually if the Geweke test is not working properly
    :param path:path of the file we want to change
    :param burn_in: addition of steps to previous burn_in index
    """
    with open(path, 'rb') as infile:
        samplefile = pickle.load(infile)
    samplefile[0].burn_in += burn_in
    with open(path, 'wb') as outfile:
        pickle.dump(samplefile, outfile)


def visualisation(mode: str = None, path: str = None, save: bool = False, savename: str = None, show: bool = False):
    """
    Visualizing sample results
    :param mode: type of visualization
    :param path: path of the sample which shall be visualized, expected to have .pickle at the end
    :param save: Whether the plot shall be saved
    :param savename: Name under which the plot shall be saved
    :param show: Whether the plot shall be presented
    """
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    with open(path, 'rb') as infile:
        samplefile = pickle.load(infile)
    if mode == 'trace':
        sample_result = result_generator(samplefile[1], samplefile[0])
        ax = visualize.sampling_fval_trace(sample_result, size=(12, 5), full_trace=True)
        if save:
            plt.savefig(fname=d + '\\plots\\' + samplefile[1] + '\\' + savename + '_trace.png')
        if show:
            plt.show()
    elif mode == '1dmarginals':
        sample_result = result_generator(samplefile[1], samplefile[0])
        ax = visualize.sampling_1d_marginals(sample_result, size=(12, 5))
        if save:
            plt.savefig(fname=d + '\\plots\\' + samplefile[1] + '\\' + savename + '_1dmarginals.png')
        if show:
            plt.show()
    elif mode == 'parameters':
        sample_result = result_generator(samplefile[1], samplefile[0])
        ax = visualize.sampling_parameters_trace(sample_result, size=(12, 5), use_problem_bounds=False, full_trace=True)
        if save:
            plt.savefig(fname=d + '\\plots\\' + samplefile[1] + '\\' + savename + '_parameters.png')
        if show:
            plt.show()


def merge_and_plot(start_sample: str = None, amount_samples: int = 10, save: bool = False, visualization: bool = False):
    """
    Merging of different sample runs into one run by cutting the burn in phase
    :param start_sample: First sample which shall be merged
    :param amount_samples: Amount of samples which shall be merged
    :param save: Whether the result shall be merged
    :param visualization: Whether the result shall be visualized
    """
    with open(start_sample, 'rb') as infile:
        result_type = pickle.load(infile)[1]
    merged_data = result_generator(result_type)
    length = 0
    data_full_sampling = [0] * amount_samples

    for n in range(amount_samples):
        with open(start_sample[:-8] + str(n) + '.pickle',
                  'rb') as infile_1:  # TODO Poblem might arise if start has double digit number 8->9
            data_full_sampling[n] = pickle.load(infile_1)[0]
            length += np.shape(data_full_sampling[n].trace_x)[1] - data_full_sampling[n].burn_in

    trace_x = np.zeros((1, length, merged_data.problem.dim))
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
        converge_size = np.shape(data_full_sampling[n].trace_x)[1] - burn_in
        merged_data.sample_result.trace_x[0, index:index + converge_size, :] \
            = data_full_sampling[n].trace_x[0, burn_in:, :]
        merged_data.sample_result.trace_neglogpost[0, index:index + converge_size] \
            = data_full_sampling[n].trace_neglogpost[0, burn_in:]
        merged_data.sample_result.trace_neglogprior[0, index:index + converge_size] \
            = data_full_sampling[n].trace_neglogprior[0, burn_in:]
        index += converge_size
    if save:
        with open(d + '\\Results_' + result_type + '\\merged_data.pickle', 'wb') as save_file:
            pickle.dump([merged_data.sample_result, result_type], save_file)

    if visualization:
        visualisation('trace', d + '\\Results_' + result_type + '\\merged_data.pickle', save=True, savename='merged')
        visualisation('1dmarginals', d + '\\Results_' + result_type + '\\merged_data.pickle', save=True,
                      savename='merged')
        visualisation('parameters', d + '\\Results_' + result_type + '\\merged_data.pickle', save=True,
                      savename='merged')


def one_dimensional_marginal(model: str = 'CR', save: bool = True):
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

    nr_params, params_fval, _, _, param_names = visualize.sampling.get_data_to_plot(result=data[0], i_chain=0,
                                                                                    stepsize=1)

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
    # plt.show()


def boxplot(mode: str = 'CPU', model: str = 'CR'):
    if model == 'CR':
        problem_1 = standard_sampling_CR()
        problem_2 = marginal_sampling_CR()
        states = 10001
        amount_samples = 50
        storage_ID = 'CR'
    elif model == 'mRNA':
        problem_1 = standard_sampling_mRNA()
        problem_2 = marginal_sampling_mRNA()
        states = 1000001
        amount_samples = 10
        storage_ID = 'mRNA'

    Result_FP = pypesto.Result(problem_1)
    Result_MP = pypesto.Result(problem_2)
    x_1 = [0.] * states
    x_2 = [0.] * states
    eff_sample_size_per_CPU = [x_1, x_2]
    CPU_time = deepcopy(eff_sample_size_per_CPU)
    for n in range(amount_samples):
        with open(d + '\\Results' + storage_ID + '_FP\\result_' + storage_ID + '_FP_' + str(n) + '.pickle', 'rb') \
            as infile_1:
            Result_FP.sample_result = pickle.load(infile_1)
            eff_sample_size_per_CPU[0][n] = pypesto.sample.effective_sample_size(Result_FP) \
                                            / Result_FP.sample_result.time
            CPU_time[0][n] = Result_FP.sample_result.time
        with open(d + '\\Results' + storage_ID + '_MP\\result_' + storage_ID + '_MP_' + str(n) + '.pickle', 'rb') \
            as infile_2:
            Result_MP.sample_result = pickle.load(infile_2)
            eff_sample_size_per_CPU[1][n] = pypesto.sample.effective_sample_size(Result_MP) \
                                            / Result_MP.sample_result.time
            CPU_time[1][n] = Result_MP.sample_result.time

    if mode == 'CPU' or mode == 'both':
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot()
        ax.boxplot(CPU_time, labels=['Full parameter', 'Marginal parameter'])
        ax.set_ylabel('CPU-time')
        plt.show()
    if mode == 'eff_ss_CPU' or mode == 'both':
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot()
        ax.boxplot(eff_sample_size_per_CPU, labels=['Full parameter', 'Marginal parameter'])
        ax.set_ylabel('Effective sample size per CPU time')
        plt.show()


def trace_plot(model: str = 'CR', sampling_type: str = 'FP', sample_selection=None, save: bool = False):
    if sample_selection is None:
        sample_selection = [0]
    if model == 'CR':
        if sampling_type == 'FP':
            problem = standard_sampling_CR()
        elif sampling_type == 'MP':
            problem = marginal_sampling_CR()
    elif model == 'mRNA':
        if sampling_type == 'FP':
            problem = standard_sampling_mRNA()
        elif sampling_type == 'MP':
            problem = marginal_sampling_mRNA()

    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    Result = pypesto.Result(problem)
    if sample_selection == 'merge':
        with open(d + '\\Results_' + model + '_' + sampling_type
                  + '\\merged_data_' + model + '_' + sampling_type + '.pickle', 'rb')as infile:
            Result.sample_result = pickle.load(infile)
        ax = visualize.sampling_fval_trace(Result, size=(12, 5), full_trace=True)
        if save:
            plt.savefig(fname=d + '\\plots\\' + model + '_' + sampling_type + '\\trace_plot_merge.png')
    else:
        for n in sample_selection:
            with open(d + '\\Results_' + model + '_' + sampling_type
                      + '\\result' + '_' + model + '_' + sampling_type + '_' + str(n) + '.pickle', 'rb')as infile:
                Result.sample_result = pickle.load(infile)
            ax = visualize.sampling_fval_trace(Result, size=(12, 5), full_trace=True)
            if save:
                plt.savefig(fname=d + '\\plots\\' + model + '_' + sampling_type + '\\' + 'trace_' + str(n) + '.png')
            # plt.show()


def parameters_plot(model: str = 'CR', sampling_type: str = 'FP', sample_selection=None, save: bool = False):
    if sample_selection is None:
        sample_selection = [0]
    if model == 'CR':
        if sampling_type == 'FP':
            problem = standard_sampling_CR()
        elif sampling_type == 'MP':
            problem = marginal_sampling_CR()
    elif model == 'mRNA':
        if sampling_type == 'FP':
            problem = standard_sampling_mRNA()
        elif sampling_type == 'MP':
            problem = marginal_sampling_mRNA()

    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    Result = pypesto.Result(problem)
    if sample_selection == 'merge':
        with open(d + '\\Results_' + model + '_' + sampling_type
                  + '\\merged_data_' + model + '_' + sampling_type + '.pickle', 'rb')as infile:
            Result.sample_result = pickle.load(infile)
        ax = visualize.sampling_parameters_trace(Result, size=(12, 5), use_problem_bounds=False, full_trace=True)
        if save:
            plt.savefig(fname=d + '\\plots\\' + model + '_' + sampling_type + '\\parameters_plot_merge.png')
    else:
        for n in sample_selection:
            with open(d + '\\Results_' + model + '_' + sampling_type
                      + '\\result' + '_' + model + '_' + sampling_type + '_' + str(n) + '.pickle', 'rb')as infile:
                Result.sample_result = pickle.load(infile)
            ax = visualize.sampling_parameters_trace(Result, size=(12, 5), use_problem_bounds=False, full_trace=True)
            if save:
                plt.savefig(fname=d + '\\plots\\' + model + '_' + sampling_type + '\\'
                                  + 'sampling_parameters_' + str(n) + '.png')
            # plt.show()


def one_d_marginals_plot(model: str = 'CR', sampling_type: str = 'FP', sample_selection=None, save: bool = False):
    if sample_selection is None:
        sample_selection = [0]
    if model == 'CR':
        if sampling_type == 'FP':
            problem = standard_sampling_CR()
        elif sampling_type == 'MP':
            problem = marginal_sampling_CR()
    elif model == 'mRNA':
        if sampling_type == 'FP':
            problem = standard_sampling_mRNA()
        elif sampling_type == 'MP':
            problem = marginal_sampling_mRNA()

    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    Result = pypesto.Result(problem)
    if sample_selection == 'merge':
        with open(d + '\\Results_' + model + '_' + sampling_type
                  + '\\merged_data_' + model + '_' + sampling_type + '.pickle', 'rb')as infile:
            Result.sample_result = pickle.load(infile)
        ax = visualize.sampling_1d_marginals(Result, size=(12, 5))
        if save:
            plt.savefig(fname=d + '\\plots\\' + model + '_' + sampling_type + '\\one_d_marginals_merge.png')
    else:
        for n in sample_selection:
            with open(d + '\\Results_' + model + '_' + sampling_type
                      + '\\result' + '_' + model + '_' + sampling_type + '_' + str(n) + '.pickle', 'rb')as infile:
                Result.sample_result = pickle.load(infile)
            ax = visualize.sampling_1d_marginals(Result, size=(12, 5))
            if save:
                plt.savefig(fname=d + '\\plots\\' + model + '_' + sampling_type + '\\'
                                  + 'one_d_marginals_' + str(n) + '.png')
            # plt.show()


def main():
    path = d + '\\Results_CR_MP\\result_CR_MP_0.pickle'
    merge_and_plot(path, 50, True, True)


main()
