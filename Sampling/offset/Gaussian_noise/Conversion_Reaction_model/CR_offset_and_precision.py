import os
import pickle
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import petab
import pypesto
import pypesto.petab
import seaborn as sns

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 17})

# path of the directory
d = os.getcwd()
# import to petab
petab_problem = petab.Problem.from_yaml(
    "conversion_reaction/SS_conversion_reaction.yaml")
tvec = np.asarray(petab_problem.measurement_df.time)
N = len(tvec)
measurement_data = np.asarray(petab_problem.measurement_df.measurement)


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


mu = 0
# std for scaling parameter --> higher = more constrained / lower = more relaxed
alpha = 100
# center the sigma parameter
beta = 0.1
# std for scaling parameter --> higher = more constrained / lower = more relaxed
kappa = 0.01


def negative_log_marginal_posterior():
    """
    dummy function for our problem function
    """
    pass


def marginal_sampling_CR():
    """Creates a pyPESTO problem."""
    objective = pypesto.Objective(fun=negative_log_marginal_posterior)
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=[-5, -5],  # lower bounds
                              ub=[5, 5],  # upper bounds
                              x_names=['$\log(k_1)$', '$\log(k_2)$'],  # parameter names
                              x_scales=['log', 'log'])  # parameter scale
    return problem


def Constant(x):
    _simulation = simulate_model(np.exp(x), tvec)
    simulation = np.asarray(_simulation)

    res = measurement_data - simulation

    summand_1 = (np.sum(res ** 2) + kappa * mu ** 2 + 2 * beta) / 2
    summand_2 = (1 / (2 * (N + kappa))) * (np.sum(res) + kappa * mu) ** 2

    return summand_1 - summand_2


def mu_(x):
    _simulation = simulate_model(np.exp(x), tvec)
    simulation = np.asarray(_simulation)

    res = measurement_data - simulation
    result_ = np.sum(res) + kappa * mu
    return result_ / (N + kappa)


def Conversion_Reaction():
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    Generator = np.random.default_rng()
    results = pypesto.Result(marginal_sampling_CR())

    with open(d + '\\Results_CR_MP\\merged_data_CR_MP.pickle', 'rb') as infile:
        results.sample_result = pickle.load(infile)[0]

    precision_list = np.zeros(np.shape(results.sample_result.trace_x)[1])
    offset_list = np.zeros(np.shape(results.sample_result.trace_x)[1])

    shape = alpha + N / 2
    for index, data in enumerate(results.sample_result.trace_x[0, :, :]):
        scale = 1 / Constant(data)  # inverse becaus the gamma sampler uses shape, scale and not alpha, beta
        precision_list[index] = Generator.gamma(shape, scale)

        new_mu = mu_(data)
        new_sigmasquare = 1 / ((N + kappa) * precision_list[index])
        offset_list[index] = Generator.normal(new_mu, new_sigmasquare)

    sns.distplot(precision_list, rug=True, axlabel='precision $\lambda$', ax=axs[1])

    sns.distplot(offset_list, rug=True, axlabel='offset $c$', ax=axs[0])

    sns.despine(fig)
    plt.tight_layout()
    plt.savefig(fname=d + '\\plots\\CR_MP\\offset_and_precision.png')

    with open('Results_CR_MP\\offset_and_precision.pickle', 'wb')as file:
        pickle.dump([offset_list, precision_list], file)


def time_calculation():
    Generator = np.random.default_rng()
    time_list = np.zeros(50)
    results = pypesto.Result(marginal_sampling_CR())
    for n in range(50):
        with open('Results_CR_MP\\result_CR_MP_' + str(n) + '.pickle', 'rb') as infile:
            results.sample_result, mode, _ = pickle.load(infile)
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
    with open('Results_CR_MP\\time_list.pickle', 'wb') as savefile:
        pickle.dump(time_list, savefile)


time_calculation()
