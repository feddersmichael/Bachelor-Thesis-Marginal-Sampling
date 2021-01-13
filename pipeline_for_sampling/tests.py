import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petab
import seaborn as sns

auto_corr = np.zeros((2, 50))

petab_problem = petab.Problem.from_yaml(
    "conversion_reaction/SS_conversion_reaction.yaml")
data = np.asarray(petab_problem.measurement_df.measurement)

a0 = 1
b0 = 0.1
k1 = np.exp(-1.55)
k2 = np.exp(-0.5)
offset = 1
precision = 950

if True:
    for n in range(50):
        # with open('Results_mRNA_MP\\result_mRNA_MP_' + str(n) + '.pickle', 'rb') as infile:
        # data_2 = pickle.load(infile)
        # auto_corr[0, n] = data_2[0].auto_correlation
        with open('Results_mRNA_FP\\result_mRNA_FP_' + str(n) + '.pickle', 'rb') as infile:
            data_3 = pickle.load(infile)
        auto_corr[1, n] = data_3[0].auto_correlation


def CR_value(t):
    res = ((k2 - k2 * np.exp(-(k2 + k1) * t)) / (k2 + k1)) + offset
    return res


time = np.linspace(0, 10, 1000)
value = np.zeros(1000)
for n in range(1000):
    value[n] = CR_value(time[n])

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()

pure_data = pd.DataFrame({"time": time, "pure_value": value})
# upper_bound_data = pd.DataFrame({"time": time, "upper_bound": (value + 3 / precision)})
# lower_bound_data = pd.DataFrame({"time": time, "lower_bound": (value - 3 / precision)})

sns.lineplot(data=pure_data, x="time", y="pure_value", ax=ax)
# sns.lineplot(data=upper_bound_data, ax=ax)
# sns.lineplot(data=lower_bound_data, ax=ax)

plt.show()
