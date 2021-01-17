import os
import pickle
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypesto
import pypesto.petab
import seaborn as sns

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 17})

# path of the directory
d = os.getcwd()

# Load experimental data
df = pd.read_csv('mRNA-transfection/data.csv', sep='\t')

tvec = np.asarray(df.Time)
N = len(tvec)
offset = 0.2
measurement_data = np.asarray(df.Measurement + offset)

mu = 0
# std for scaling parameter --> higher = more constrained / lower = more relaxed
alpha = 100
# center the sigma parameter
beta = 0.1
# std for scaling parameter --> higher = more constrained / lower = more relaxed
kappa = 0.01


def analytical_x2(t, t0, kTL_m0, xi, delta):
    X = [np.exp(-delta * (t - t0)) * (t > t0),
         kTL_m0 * (np.exp(-xi * (t - t0)) - np.exp(-delta * (t - t0))) / (delta - xi) * (t > t0)]
    return X[1]


def simulate_model(x, tvec):
    # assign parameters
    t0, kTL_m0, xi, delta = x
    # simulate model
    simulation = np.asarray([analytical_x2(t, t0, kTL_m0, xi, delta)
                             for t in tvec])
    return simulation


def negative_log_marginal_posterior():
    """
    dummy function for our problem function
    """
    pass


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


def Constant(x):
    _simulation = simulate_model(np.power(10, x), tvec)
    simulation = np.asarray(_simulation)

    res = measurement_data - simulation

    summand_1 = (np.sum(res ** 2) + kappa * mu ** 2 + 2 * beta) / 2
    summand_2 = (1 / (2 * (N + kappa))) * (np.sum(res) + kappa * mu) ** 2

    return summand_1 - summand_2


def mu_(x):
    simulation = simulate_model(np.power(10, x), tvec)

    res = measurement_data - simulation
    result_ = np.sum(res) + kappa * mu
    return result_ / (N + kappa)


def mRNA_transfection():
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    Generator = np.random.default_rng()
    results = pypesto.Result(marginal_sampling_mRNA())

    with open(d + '\\Results_mRNA_MP\\merged_data_mRNA_MP.pickle', 'rb') as infile:
        results.sample_result = pickle.load(infile)[0]

    precision_list = np.zeros(np.shape(results.sample_result.trace_x)[1])
    offset_list = np.zeros(np.shape(results.sample_result.trace_x)[1])

    shape = alpha + N / 2
    for index, data in enumerate(results.sample_result.trace_x[0, :, :]):
        if index % 1000000 == 0:
            print(index)

        scale = 1 / Constant(data)  # inverse because the gamma sampler uses shape, scale and not alpha, beta
        precision_list[index] = Generator.gamma(shape, scale)

    print('starting precision plot')
    sns.distplot(precision_list, rug=True, axlabel='precision', ax=axs[1])
    print('precision plot done')

    for index, data in enumerate(results.sample_result.trace_x[0, :, :]):
        if index % 1000000 == 0:
            print(index)
        new_mu = mu_(data)
        new_sigmasquare = 1 / ((N + kappa) * precision_list[index])
        offset_list[index] = Generator.normal(new_mu, new_sigmasquare)

    print('starting offset plot')
    sns.distplot(offset_list, rug=True, axlabel='offset', ax=axs[0])
    print('precision offset done')

    sns.despine(fig)
    plt.tight_layout()
    plt.savefig(fname=d + '\\plots\\mRNA_MP\\offset_and_precision.png')

    with open('Results_mRNA_MP\\offset_and_precision.pickle', 'wb')as file:
        pickle.dump([offset_list, precision_list], file)


def time_calculation():
    Generator = np.random.default_rng()
    results = pypesto.Result(marginal_sampling_mRNA())

    for n in range(50):
        print(n)
        with open('Results_mRNA_MP\\time_list.pickle', 'rb') as infile:
            time_list = pickle.load(infile)
        with open('Results_mRNA_MP\\result_mRNA_MP_' + str(n) + '.pickle', 'rb') as infile:
            results.sample_result, mode = pickle.load(infile)
        precision_list = np.zeros(np.shape(results.sample_result.trace_x)[1])
        offset_list = np.zeros(np.shape(results.sample_result.trace_x)[1])
        shape = alpha + N / 2

        start_time = process_time()
        for index, data in enumerate(results.sample_result.trace_x[0, :, :]):
            scale = 1 / Constant(data)  # inverse becaus the gamma sampler uses shape, scale and not alpha, beta
            precision_list[index] = Generator.gamma(shape, scale)

            new_mu = mu_(data)
            new_sigmasquare = 1 / ((N + kappa) * precision_list[index])
            offset_list[index] = Generator.normal(new_mu, new_sigmasquare)

        duration = process_time() - start_time
        time_list[n] = duration
        print(duration)
        with open('Results_mRNA_MP\\time_list.pickle', 'rb') as savefile:
            pickle.dump(time_list, savefile)


time_calculation()
