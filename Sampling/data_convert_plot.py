"""
Several functions to convert or plot results from our sample process
"""

import os
import pickle
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petab
import pypesto
import pypesto.petab
import pypesto.sample as sample
import pypesto.visualize as visualize
import seaborn as sns

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 17})

d = os.getcwd()

# estimated values in mRNA, linear scale
t_0 = 1.9939041943340294
kTLm_0 = 9.882089134453418
xi = [0.7804683225514181, 0.20374850385622745]
delta = [0.20375774544786057, 0.7804071490541745]
offset_mRNA = 0.19856558250094725
precision_mRNA = 299.8008024988325
# estimated values in CR, linear scale
k_1 = 0.20669033739193668
k_2 = 0.5957098364375092
offset_CR = 1.0148049568031505
precision_CR = 975.4361871157674


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


def X_2_CR(t, a0=1, b0=0, k1=k_1, k2=k_2):
    """
    analytical solution to the X_2 parameter in the ODE model for the CR model
    :param t:
    :param a0:
    :param b0:
    :param k1:
    :param k2:
    :return:
    """
    return (k2 - k2 * np.exp(-(k2 + k1) * t)) / (k2 + k1)


def X_2_mRNA(t, t0=t_0, kTL_m0=kTLm_0, xi_=xi[0], delta_=delta[0]):
    """
    analytical solution to the X_2 parameter in the ODE model for the mRNA model
    :param delta_:
    :param xi_:
    :param t:
    :param t0:
    :param kTL_m0:
    :return:
    """
    X = [np.exp(-delta_ * (t - t0)) * (t > t0),
         kTL_m0 * (np.exp(-xi_ * (t - t0)) - np.exp(-delta_ * (t - t0))) / (delta_ - xi_) * (t > t0)]
    return X[1]


def Gaussian_noise_CR_model_FP_sampling():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5, -np.inf, 0],  # lower bounds
                              ub=[5, 5, np.inf, np.inf],  # upper bounds
                              x_names=['$\log(k_1)$', '$\log(k_2)$', 'offset $c$', 'precision $\lambda$'],
                              # parameter names
                              x_scales=['log', 'log', 'lin', 'lin'])  # parameter scale
    return problem


def Laplacian_noise_CR_model_FP_sampling():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5, 0, 0],  # lower bounds
                              ub=[5, 5, np.inf, np.inf],  # upper bounds
                              x_names=['$\log(k_1)$', '$\log(k_2)$', 'offset $c$', 'scale $\sigma$'],
                              # parameter names
                              x_scales=['log', 'log', 'lin', 'lin'])  # parameter scale
    return problem


def marginal_sampling_CR():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_marginal_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5],  # lower bounds
                              ub=[5, 5],  # upper bounds
                              x_names=['$\log(k_1)$', '$\log(k_2)$'],  # parameter names
                              x_scales=['log', 'log'])  # parameter scale
    return problem


def standard_sampling_mRNA():
    """Creates a pyPESTO problem."""
    df = pd.read_csv('mRNA-transfection/data.csv', sep='\t')
    objective = pypesto.Objective(fun=negative_log_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-2, -5, -5, -5, -np.inf, 0],  # lower bounds
                              ub=[np.log10(df.Time.max()), 5, 5, 5, np.inf, np.inf],
                              # upper bounds 't', 'k', 'xi, 'delta'
                              x_names=['$\log_{10}(t_1)$', '$\log_{10}(k_{TL} \cdot m_1)$', '$\log_{10}(\\xi)$',
                                       '$\log_{10}(\delta)$', 'offset $c$', 'precision $\lambda$'],  # parameter names
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
                              x_names=['$\log_{10}(t_1)$', '$\log_{10}(k_{TL} \cdot m_1)$', '$\log_{10}(\\xi)$',
                                       '$\log_{10}(\delta)$'],  # parameter names
                              x_scales=['log10', 'log10', 'log10', 'log10'])  # parameter scale
    return problem


def result_generator(result_type: list = None, sample_result: str = None):
    """
    Transforms a sample_result into a proper Result object
    :param result_type: Defines the object function we use
    :param sample_result: if given already includes the sample_result into the new Result object
    :return: the generated Result object
    """
    if result_type[0] == 'Gaussian':
        if result_type[1] == 'Conversion_Reaction':
            if result_type[2] == 'Full_parameter':
                generated_result = pypesto.Result(Gaussian_noise_CR_model_FP_sampling())
                if sample_result is not None:
                    generated_result.sample_result = sample_result
                return generated_result
            else:
                generated_result = pypesto.Result(marginal_sampling_CR())
                if sample_result is not None:
                    generated_result.sample_result = sample_result
                return generated_result
        if result_type[1] == 'mRNA':
            if result_type[2] == 'Full_parameter':
                generated_result = pypesto.Result(standard_sampling_mRNA())
                if sample_result is not None:
                    generated_result.sample_result = sample_result
                return generated_result
            else:
                generated_result = pypesto.Result(marginal_sampling_mRNA())
                if sample_result is not None:
                    generated_result.sample_result = sample_result
                return generated_result
    if result_type[0] == 'Laplacian':
        if result_type[1] == 'Conversion_Reaction':
            if result_type[2] == 'Full_parameter':
                generated_result = pypesto.Result(Laplacian_noise_CR_model_FP_sampling())
                if sample_result is not None:
                    generated_result.sample_result = sample_result
                return generated_result


def burn_in_change(path: str = None, burn_in: int = None):
    """
    Changes the burn_in parameter manually if the Geweke test is not working properly
    :param path:path of the file we want to change
    :param burn_in: new burn in index
    """
    with open(path, 'rb') as infile:
        samplefile = pickle.load(infile)
    samplefile[0].burn_in = burn_in
    with open(path, 'wb') as outfile:
        pickle.dump(samplefile, outfile)


def visualisation(mode: str, path: str, save: bool = False, savename: str = None, show: bool = False):
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
    sample_result = result_generator([samplefile[1], samplefile[2], samplefile[3]], samplefile[0])
    if mode == 'trace':
        ax = visualize.sampling_fval_trace(sample_result, size=(12, 5), full_trace=True)
        if show:
            plt.show()
        if save:
            plt.savefig(fname=samplefile[0] + '_noise/' + samplefile[1] + '_model/Results/' + samplefile[2]
                              + '/' + savename + '.png')
    elif mode == '1d_marginals':
        ax = visualize.sampling_1d_marginals(sample_result, size=(12, 5))
        if show:
            plt.show()
        if save:
            plt.savefig(fname=samplefile[0] + '_noise/' + samplefile[1] + '_model/Results/' + samplefile[2]
                              + '/' + savename + '.png')
    elif mode == 'parameters':
        ax = visualize.sampling.sampling_parameters_trace(sample_result, size=(12, 5), use_problem_bounds=False,
                                                          full_trace=True)
        if show:
            plt.show()
        if save:
            plt.savefig(fname=samplefile[0] + '_noise/' + samplefile[1] + '_model/Results/' + samplefile[2]
                              + '/' + savename + '.png')


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
                  'rb') as infile_1:  # TODO Problem might arise if start has double digit number 8->9
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
        with open(d + '\\Results_' + result_type + '\\merged_data_' + result_type + '.pickle', 'wb') as save_file:
            pickle.dump([merged_data.sample_result, result_type], save_file)

    if visualization:
        print('trace')
        visualisation('trace', d + '\\Results_' + result_type + '\\merged_data_' + result_type + '.pickle', save=True,
                      savename='merged_trace_' + result_type)
        print('1d_marginals')
        visualisation('1d_marginals', d + '\\Results_' + result_type + '\\merged_data_' + result_type + '.pickle',
                      save=True, savename='merged_1d_marginals_' + result_type)
        print('parameters')
        visualisation('parameters', d + '\\Results_' + result_type + '\\merged_data_' + result_type + '.pickle',
                      save=True, savename='merged_parameters_' + result_type)


def overlap_plot(model: str = 'CR', save: bool = True):
    """
    Creates overlapping 1d_marginal plot
    :param model: Which model we use
    :param save: Whether the result shall be saved
    """
    if model == 'CR':
        path_FP = d + '\\Results_CR_FP\\merged_data_CR_FP.pickle'
        path_MP_1 = d + '\\Results_CR_MP\\merged_data_CR_MP.pickle'
        path_MP_2 = d + '\\Results_CR_MP\\offset_and_precision.pickle'
    elif model == 'mRNA':
        path_FP = d + '\\Results_mRNA_FP\\merged_data_mRNA_FP.pickle'
        path_MP_1 = d + '\\Results_mRNA_MP\\merged_data_mRNA_MP.pickle'
        path_MP_2 = d + '\\Results_mRNA_MP\\offset_and_precision.pickle'

    data_samples = [0, 0]
    with open(path_FP, 'rb') as data_file:
        samplefile = pickle.load(data_file)
        data_samples[0] = result_generator(samplefile[1], samplefile[0])

    with open(path_MP_1, 'rb') as data_file:
        samplefile = pickle.load(data_file)
        data_samples[1] = result_generator(samplefile[1], samplefile[0])

    nr_params, params_fval, _, _, param_names = visualize.sampling.get_data_to_plot(result=data_samples[0], i_chain=0,
                                                                                    stepsize=1)

    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    fig, ax = plt.subplots(num_row, num_col, squeeze=False, figsize=(12, 5))

    par_ax = dict(zip(param_names, ax.flat))
    sns.set(style="ticks")

    for idx, par_id in enumerate(param_names):
        print(idx)
        if idx != 0:
            sns.distplot(params_fval[par_id], rug=True, ax=par_ax[par_id])

            par_ax[par_id].set_xlabel(param_names[idx])
            par_ax[par_id].set_ylabel('Density')
        else:
            sns.distplot(params_fval[par_id], rug=True, ax=par_ax[par_id], label='FP-approach')

            par_ax[par_id].set_xticks([0.297, 0.300, 0.303])
            par_ax[par_id].set_xlabel(param_names[idx])
            par_ax[par_id].set_ylabel('Density')

    _, params_fval, _, _, param_names = visualize.sampling.get_data_to_plot(result=data_samples[1], i_chain=0,
                                                                            stepsize=1)

    for idx, par_id in enumerate(param_names):
        print(idx)
        sns.distplot(params_fval[par_id], rug=True, ax=par_ax[par_id])
        if idx == 0:
            par_ax[par_id].set_xticks([0.297, 0.300, 0.303])

    with open(path_MP_2, 'rb') as file:
        samples = pickle.load(file)
    print('offset')
    sns.distplot(samples[0], rug=True, ax=par_ax['offset $c$'], label='MP-approach')
    print('precision')
    sns.distplot(samples[1], rug=True, ax=par_ax['precision $\lambda$'])

    sns.despine()
    fig.tight_layout()
    fig.legend(bbox_to_anchor=(0.5, -0.03), loc='upper center', mode='expand', ncol=2, borderaxespad=0)
    if save:
        plt.savefig(fname='plots\\combination\\' + model + '\\overlap_plot_' + model + '.png', bbox_inches="tight")


def single_parameter_visualization():
    x = 0


def boxplot_mRNA_auto():
    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
    path = ['Results_mRNA_FP\\result_mRNA_FP_']

    times_FP_conv = np.zeros(4)
    times_FP_slight = np.zeros(4)
    times_FP_no_conv = np.zeros(32)
    autocorr_FP_conv = np.zeros(4)
    autocorr_FP_slight = np.zeros(4)
    autocorr_FP_no_conv = np.zeros(32)

    FP_converges = [2, 20, 23, 37]
    FP_slight_convergence = [14, 21, 28, 30]
    FP_no_convergence = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27, 29, 31, 32,
                         33, 34, 35, 36, 38, 39]

    for n, values in enumerate(FP_converges):
        with open(path[0] + str(values) + '.pickle', 'rb') as infile:
            sample = pickle.load(infile)
        times_FP_conv[n] = sample[0].time
        autocorr_FP_conv[n] = sample[0].auto_correlation

    for n, values in enumerate(FP_slight_convergence):
        with open(path[0] + str(values) + '.pickle', 'rb') as infile:
            sample = pickle.load(infile)
        times_FP_slight[n] = sample[0].time
        autocorr_FP_slight[n] = sample[0].auto_correlation

    for n, values in enumerate(FP_no_convergence):
        with open(path[0] + str(values) + '.pickle', 'rb') as infile:
            sample = pickle.load(infile)
        times_FP_no_conv[n] = sample[0].time
        autocorr_FP_no_conv[n] = sample[0].auto_correlation

    times = [times_FP_no_conv, times_FP_slight, times_FP_conv]
    autocorr = [autocorr_FP_no_conv, autocorr_FP_slight, autocorr_FP_conv]

    axs[0].boxplot(times, labels=['one mode', 'two modes temporary', 'convergence'])
    axs[0].set_ylabel('CPU-time in seconds')
    axs[0].set_yticks([0, 1000, 2000, 3000, 4000])

    axs[1].boxplot(autocorr, labels=['one mode', 'two modes temporary', 'convergence'])
    axs[1].set_ylabel('auto-correlation')

    plt.show()


def boxplot_mRNA(mode: str = 'theta'):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    path = ['Results_mRNA_FP\\result_mRNA_FP_', 'Results_mRNA_MP\\result_mRNA_MP_']
    if mode == 'all':
        with open('Results_mRNA_MP\\time_list.pickle', 'rb') as infile:
            extra_time = pickle.load(infile)

    times_FP = np.zeros(4)
    times_MP = np.zeros(10)
    effective_SS_FP = np.zeros(4)
    effective_SS_MP = np.zeros(10)

    FP_converges = [2, 20, 23, 37]

    for n, values in enumerate(FP_converges):
        with open(path[0] + str(values) + '.pickle', 'rb') as infile:
            sample = pickle.load(infile)
        times_FP[n] = sample[0].time
        effective_SS_FP[n] = sample[0].effective_sample_size

    for n in range(10):
        with open(path[1] + str(n) + '.pickle', 'rb') as infile:
            sample = pickle.load(infile)
        if mode == 'theta':
            times_MP[n] = sample[0].time
        elif mode == 'all':
            times_MP[n] = sample[0].time + extra_time[n]
        effective_SS_MP[n] = sample[0].effective_sample_size

    times = [times_FP, times_MP]

    effect_per_time = [np.divide(effective_SS_FP, times_FP), np.divide(effective_SS_MP, times_MP)]

    axs[0].boxplot(times, labels=['FP-approach', 'MP-approach'])
    axs[0].set_ylabel('CPU-time in seconds')
    axs[0].set_yticks([0, 1000, 2000, 3000, 4000, 5000])

    axs[1].boxplot(effect_per_time, labels=['FP-approach', 'MP-approach'])
    axs[1].set_ylabel('Effective sample size per CPU-time')

    plt.show()


def boxplot_CR(mode: str = 'theta'):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    amount_runs = 50
    path = ['Results_CR_FP\\result_CR_FP_', 'Results_CR_MP\\result_CR_MP_']
    if mode == 'all':
        with open('Results_CR_MP\\time_list.pickle', 'rb') as infile:
            extra_time = pickle.load(infile)

    times_FP = np.zeros(amount_runs)
    times_MP = np.zeros(amount_runs)
    effective_SS_FP = np.zeros(amount_runs)
    effective_SS_MP = np.zeros(amount_runs)

    for n in range(amount_runs):
        with open(path[0] + str(n) + '.pickle', 'rb') as infile:
            sample = pickle.load(infile)
        times_FP[n] = sample[0].time
        effective_SS_FP[n] = sample[0].effective_sample_size

        with open(path[1] + str(n) + '.pickle', 'rb') as infile:
            sample = pickle.load(infile)
        if mode == 'theta':
            times_MP[n] = sample[0].time
        elif mode == 'all':
            times_MP[n] = sample[0].time + extra_time[n]
        effective_SS_MP[n] = sample[0].effective_sample_size

    times = [times_FP, times_MP]
    effect_per_time = [np.divide(effective_SS_FP, times_FP), np.divide(effective_SS_MP, times_MP)]

    axs[0].boxplot(times, labels=['FP-approach', 'MP-approach'])
    axs[0].set_ylabel('CPU-time in seconds')
    axs[0].set_yticks([0, 1, 2, 3, 4, 5])

    axs[1].boxplot(effect_per_time, labels=['FP-approach', 'MP-approach'])
    axs[1].set_ylabel('Effective sample size per CPU-time')

    plt.show()


def parameter_estimation_mRNA():
    """
    calculating the median for all sampled values after the burn in for the mRNA model
    :return: the two possibilities for the parameters
    """
    with open('Results_mRNA_MP\\merged_data_mRNA_MP.pickle', 'rb') as infile:
        trace = pickle.load(infile)[0].trace_x[0, :, :]
        t_0_list = trace[:, 0]
        kTLm_O_list = trace[:, 1]
        xi_list = trace[:, 2]
        delta_list = trace[:, 3]
    with open('Results_mRNA_MP\\offset_and_precision.pickle', 'rb') as infile:
        offset_list, precision_list = pickle.load(infile)

    offset = statistics.median(offset_list)
    precision = statistics.median(precision_list)
    t_0 = statistics.median(t_0_list)
    kTLm_O = statistics.median(kTLm_O_list)
    xi_1 = []
    xi_2 = []
    delta_1 = []
    delta_2 = []
    for n, xi in enumerate(xi_list):
        delta = delta_list[n]
        if -0.2 < xi < 0:
            xi_1.append(xi)
            delta_1.append(delta)
        elif -0.8 < xi < -0.6:
            xi_2.append(xi)
            delta_2.append(delta)
    xi_1 = statistics.median(xi_1)
    xi_2 = statistics.median(xi_2)
    delta_1 = statistics.median(delta_1)
    delta_2 = statistics.median(delta_2)
    return [[t_0, kTLm_O, xi_1, delta_1, offset, precision], [t_0, kTLm_O, xi_2, delta_2, offset, precision]]


def parameter_estimation_CR():
    """
    calculating the median for all sampled values after the burn in for the CR model
    :return: the list of the calculated parameters
    """
    with open('Results_CR_MP\\merged_data_CR_MP.pickle', 'rb') as infile:
        trace = pickle.load(infile)[0].trace_x[0, :, :]
        k_1_list = trace[:, 0]
        k_2_list = trace[:, 1]
    with open('Results_CR_MP\\offset_and_precision.pickle', 'rb') as infile:
        offset_list, precision_list = pickle.load(infile)

    offset = statistics.median(offset_list)
    precision = statistics.median(precision_list)
    k_1 = statistics.median(k_1_list)
    k_2 = statistics.median(k_2_list)
    return [k_1, k_2, offset, precision]


def data_sample_comparison(mode: str = 'CR'):
    if mode == 'CR':
        fig = plt.figure(figsize=(12, 5))
        ax = plt.subplot()
        petab_problem = petab.Problem.from_yaml(
            "conversion_reaction/SS_conversion_reaction.yaml")
        data = np.asarray(petab_problem.measurement_df.measurement)
        tvec = np.asarray(petab_problem.measurement_df.time)
        x = np.linspace(0, 10, 1001)
        function = np.zeros(1001)
        for n, value in enumerate(x):
            function[n] = X_2_CR(value) + offset_CR
        df = pd.DataFrame({'time $t$': x, '$X_2$': function})
        ax = sns.lineplot(x="time $t$", y="$X_2$", data=df, label='estimated function')
        sigma = np.sqrt(1 / precision_CR)
        upper_bound = function + 3 * sigma
        lower_bound = function - 3 * sigma
        ax.fill_between(x, lower_bound, upper_bound, alpha=0.1, label='3$\sigma$ confidence interval')
        sns.scatterplot(x=tvec, y=data, palette='orange', ax=ax, label='measured data')
        ax.legend(loc=4)
        plt.show()

    if mode == 'mRNA':
        fig = plt.figure(figsize=(12, 5))
        ax = plt.subplot()
        df_measurement = pd.read_csv('mRNA-transfection/data.csv', sep='\t')
        df_measurement.Measurement += 0.2
        x = np.linspace(0, 10, 1001)
        function = np.zeros(1001)
        for n, value in enumerate(x):
            function[n] = X_2_mRNA(value) + offset_mRNA
        df = pd.DataFrame({'time $t$': x, '$X_2$': function})
        ax = sns.lineplot(x="time $t$", y="$X_2$", data=df, label='estimated function')
        sigma = np.sqrt(1 / precision_mRNA)
        upper_bound = function + 3 * sigma
        lower_bound = function - 3 * sigma
        ax.fill_between(x, lower_bound, upper_bound, alpha=0.1, label='3$\sigma$ confidence interval')
        sns.scatterplot(x=df_measurement.Time, y=df_measurement.Measurement, palette='orange', ax=ax,
                        label='measured data')
        ax.legend(loc=4)
        plt.show()


def main():
    """
    Main
    """
    visualisation('parameters',
                  'offset/Laplacian_noise/Conversion_Reaction_model/exponential_prior/Results/Full_parameter/standard_choice.pickle',
                  True, 'standard_choice')


main()
