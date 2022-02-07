#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pypesto
import pypesto.petab
import pypesto.sample as sample
import pypesto.visualize as visualize
from scipy.special import gammaln

import os
import sys
import petab
import amici
import numpy as np
from copy import deepcopy


# In[2]:


# import to petab
petab_problem = petab.Problem.from_yaml(
    "Sampling/offset_and_scaling/Borghans_BiophysChem1997/Borghans_BiophysChem1997.yaml"
)
# import to pypesto
importer = pypesto.petab.PetabImporter(petab_problem)

# create problem
problem = importer.create_problem()


def negative_log_prior(x):

    scaling = x[19]
    offset = x[18]
    precision = x[20]

    precision_prior = alpha * np.log(beta) - gammaln(alpha) + (alpha - 1) * np.log(precision) - beta * precision

    scaling_prior = 0.5 * (np.log(precision) + np.log(tau) - np.log(2) - np.log(np.pi)) - (precision * tau * 0.5 * (scaling - z)**2)

    offset_prior = 0.5 * (np.log(precision) + np.log(kappa) - np.log(2) - np.log(np.pi)) - (precision * kappa * 0.5 * (offset - mu)**2)

    return - (precision_prior + scaling_prior + offset_prior)

#objective_function = pypesto.

# In[3]:


x_full = np.asarray(deepcopy(petab_problem.x_nominal_scaled))

res = problem.objective.call_unprocessed(x_full, [1], 'mode_fun')

print(res['fval'])


# In[4]:


# create edatas
edatas = importer.create_edatas()

# measurement data
_data = [amici.numpy.ExpDataView(edata)['observedData']
         for edata in edatas]
data = np.concatenate(_data, axis=0)

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


import matplotlib.pyplot as plt

plt.figure()
plt.plot(res['rdatas'][0].t,res['rdatas'][0].y,'r-',label='Model simulation')
plt.plot(res['rdatas'][0].t,data,'.k',label='Experimental data')
plt.ylim([0.2,1.2])
plt.legend()
plt.show()


# In[6]:


sampler = sample.AdaptiveMetropolisSampler()
n_samples = 20000
x0s = x_full


# In[ ]:

np.random.seed(0)
result = sample.sample(problem, n_samples=n_samples,
                       sampler=sampler, x0=x0s)
sample.effective_sample_size(result)


# In[ ]:


ax = visualize.sampling_1d_marginals(result, size=(13,13))

plt.show()
# In[ ]:


ax = visualize.sampling_fval_traces(result,full_trace=True, size=(12,3))

plt.show()
# In[ ]:


idx_MAP = np.argmax(-result.sample_result.trace_neglogpost[0][result.sample_result.burn_in:])

x_MAP = result.sample_result.trace_x[0][result.sample_result.burn_in+idx_MAP,:]

res_MAP = problem.objective.call_unprocessed(x_MAP, [0], 'mode_fun')


# In[ ]:


plt.figure()
plt.plot(res['rdatas'][0].t,res['rdatas'][0].y,'b-',label='Nominal model simulation')
plt.plot(res_MAP['rdatas'][0].t,res_MAP['rdatas'][0].y,'r-',label='MAP model simulation')
plt.plot(res_MAP['rdatas'][0].t,data,'.k',label='Experimental data')
plt.ylim([0.2,1.2])
plt.legend()
plt.show()


# In[ ]:




