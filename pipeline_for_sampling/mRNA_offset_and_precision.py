import pypesto
import pypesto.petab
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
import os

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


def marginal_sampling():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_marginal_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-2, -5, -5, -5],  # lower bounds
                              ub=[np.log10(df.Time.max()), 5, 5, 5],  # upper bounds
                              x_names=['t_0', 'k_{TL}*m_0', 'xi', 'delta'],  # parameter names
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
    results = pypesto.Result(marginal_sampling())

    with open(d + '\\Results_mRNA_MP\\merged_data_mRNA_MP.pickle', 'rb') as infile:
        results.sample_result = pickle.load(infile)[0]

    precision_list = np.zeros(np.shape(results.sample_result.trace_x)[1])

    for index, data in enumerate(results.sample_result.trace_x[0, :, :]):
        if index % 1000000 == 0:
            print(index)
        shape = alpha + N / 2
        scale = 1 / Constant(data)  # inverse because the gamma sampler uses shape, scale and not alpha, beta
        precision_list[index] = Generator.gamma(shape, scale)

    print('starting precision plot')
    sns.distplot(precision_list, rug=True, axlabel='precision', ax=axs[1])
    print('precision plot done')

    offset_list = np.zeros(np.shape(results.sample_result.trace_x)[1])

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


mRNA_transfection()
