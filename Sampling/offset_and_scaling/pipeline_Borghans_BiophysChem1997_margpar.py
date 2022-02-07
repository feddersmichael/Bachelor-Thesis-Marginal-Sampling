#!/usr/bin/env python
# coding: utf-8

# In[1]:


from copy import deepcopy

import amici
import numpy as np
import petab
import pypesto.petab
import pypesto.sample as sample
import pypesto.visualize as visualize
from scipy.special import gammaln
import os

# In[2]:

location = os.getcwd()
print(location)
# import to petab
petab_problem = petab.Problem.from_yaml(
    "Sampling/offset_and_scaling/Borghans_marginalised/Borghans_BiophysChem1997.yaml"
)
# import to pypesto
importer = pypesto.petab.PetabImporter(petab_problem)

# create problem
_problem = importer.create_problem()

lb = deepcopy(_problem.lb_full)
ub = deepcopy(_problem.ub_full)
x_scales = deepcopy(_problem.x_scales)
x_names = deepcopy(_problem.x_names)

# In[3]:


# create edatas
edatas = importer.create_edatas()

# measurement data
_data = [amici.numpy.ExpDataView(edata)['observedData']
         for edata in edatas]
data = np.concatenate(_data, axis=0)

# In[4]:

mu = 0
z = 0
# std for scaling parameter --> higher = more constrained / lower = more relaxed
alpha = 100
# center the sigma parameter
beta = 0.1
# std for scaling parameter --> higher = more constrained / lower = more relaxed
kappa = 0.01
tau = 0.01


# In[5]:

def negative_log_marginal_posterior(x):
    # experimental data
    # data = np.random.rand(10)

    N = data.shape[0]

    _simulation = _problem.objective.call_unprocessed(x, [0], 'mode_fun')
    simulation = _simulation['rdatas'][0].y

    C_1 = kappa * mu ** 2 + tau * z ** 2 + 2 * beta + np.sum(data ** 2) - (kappa * mu + np.sum(data)) ** 2 / (N + kappa)

    C_2 = ((kappa * mu + np.sum(data)) * np.sum(simulation) - (N + kappa) * (tau * z + np.sum(simulation * data))) ** 2

    C_3 = (N + kappa) * ((N + kappa) * (tau + np.sum(simulation ** 2)) - np.sum(simulation) ** 2)

    logC = np.log(C_1 - C_2 / C_3) - np.log(2)

    mL_1 = alpha * (np.log(beta) - logC) - (
            gammaln(alpha) + (N / 2) * (logC + np.log(2) + np.log(np.pi))) + 0.5 * (
                   np.log(kappa) + np.log(tau))

    mL_2 = -0.5 * np.log(
        (N + kappa) * (tau + np.sum(simulation ** 2)) - np.sum(simulation) ** 2) + gammaln(alpha + N / 2)

    return -(mL_1 + mL_2)


x_full = np.asarray(deepcopy(petab_problem.x_nominal_scaled))

res = _problem.objective.call_unprocessed(x_full, [1], 'mode_fun')

print(res['fval'])

marg = negative_log_marginal_posterior(x_full)
print(marg)

# In[6]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(res['rdatas'][0].t, res['rdatas'][0].y, 'r-', label='Model simulation')
plt.plot(res['rdatas'][0].t, data, '.k', label='Experimental data')
plt.ylim([0, 1.2])
plt.legend()
plt.show()

objective_function = pypesto.Objective(fun=negative_log_marginal_posterior)

problem = pypesto.Problem(objective=objective_function, lb=lb, ub=ub, x_scales=x_scales, x_names=x_names)

# In[7]:

np.random.seed(0)
sampler = sample.AdaptiveMetropolisSampler()
n_samples = 500000
x0s = x_full

# In[8]:


result = sample.sample(problem, n_samples=n_samples,
                       sampler=sampler, x0=x0s)
sample.effective_sample_size(result)

# In[9]:


ax = visualize.sampling_1d_marginals(result, size=(13, 13))
plt.show()
# In[10]:
ax = visualize.sampling_fval_traces(result, full_trace=True, size=(12, 3))
plt.show()

# In[10]:
ax = visualize.sampling_parameter_traces(result, full_trace=True, size=(12, 3))
plt.show()
# In[11]:


idx_MAP = np.argmax(-result.sample_result.trace_neglogpost[0][result.sample_result.burn_in:])

x_MAP = result.sample_result.trace_x[0][result.sample_result.burn_in + idx_MAP, :]

res_MAP = _problem.objective.call_unprocessed(x_MAP, [0], 'mode_fun')

# In[12]:


plt.figure()
plt.plot(res['rdatas'][0].t, res['rdatas'][0].y, 'b-', label='Nominal model simulation')
plt.plot(res_MAP['rdatas'][0].t, res_MAP['rdatas'][0].y, 'r-', label='MAP model simulation')
plt.plot(res_MAP['rdatas'][0].t, data, '.k', label='Experimental data')
plt.ylim([0.2, 1.2])
plt.legend()
plt.show()
