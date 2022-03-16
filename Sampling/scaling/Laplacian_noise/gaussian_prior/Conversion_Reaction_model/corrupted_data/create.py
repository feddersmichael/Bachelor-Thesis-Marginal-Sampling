import numpy as np
import pandas as pd
import petab
from petab.C import *

a0 = 1
b0 = 0.01
k1 = 1
k2 = 0.1
sc = 0.9


def analytical_b(t, a0=a0, b0=b0, k1=k1, k2=k2):
    return (k1 - k1 * np.exp(-(k1 + k2) * t)) / (k1 + k2)


# problem --------------------------------------------------------------------

condition_df = pd.DataFrame(data={
    CONDITION_ID: ['c0'],
}).set_index([CONDITION_ID])

times = np.asarray(range(11))
nt = len(times)
simulations = [analytical_b(t, a0=a0, b0=b0, k1=k1, k2=k2)
               for t in times]
sigma = 0.015
# measurements = (sc + np.asarray(simulations)) + sigma * np.random.randn(nt)
# measurements = measurements.clip(0.)

offset = 1

measurements = np.asarray([0.0219, 0.4442, 0.5840, 0.6951, 0.7249, 0.7242, 0.7398, 0.7506, 0.7843, 0.7768, 0.7359])
measurements += offset

measurement_df = pd.DataFrame(data={
    OBSERVABLE_ID: ['obs_b'] * nt,
    SIMULATION_CONDITION_ID: ['c0'] * nt,
    TIME: times,
    MEASUREMENT: measurements,
    OBSERVABLE_PARAMETERS: ['offset'] * nt,
    NOISE_PARAMETERS: ['sigma'] * nt
})

observable_df = pd.DataFrame(data={
    OBSERVABLE_ID: ['obs_b'],
    OBSERVABLE_FORMULA: ['observableParameter1_obs_b + B'],
    NOISE_FORMULA: ['noiseParameter1_obs_b'],
    OBSERVABLE_TRANSFORMATION: [LOG]
}).set_index([OBSERVABLE_ID])

parameter_df = pd.DataFrame(data={
    PARAMETER_ID: ['k1', 'k2', 'offset', 'sigma'],
    PARAMETER_SCALE: [LOG] * 4,
    LOWER_BOUND: [0.0067, 0.0067, 1e-12, 1e-12],
    UPPER_BOUND: [148.413, 148.413, 1e12, 1e12],
    NOMINAL_VALUE: [k1, k2, sc, sigma],
    ESTIMATE: [1, 1, 1, 1],
}).set_index(PARAMETER_ID)

petab.write_condition_df(condition_df, "SS_conditions.tsv")
petab.write_measurement_df(measurement_df, "SS_measurements.tsv")
petab.write_observable_df(observable_df, "SS_observables.tsv")
petab.write_parameter_df(parameter_df, "SS_parameters.tsv")

yaml_config = {
    FORMAT_VERSION: 1,
    PARAMETER_FILE: "SS_parameters.tsv",
    PROBLEMS: [{
        SBML_FILES: ["model_conversion_reaction.xml"],
        CONDITION_FILES: ["SS_conditions.tsv"],
        MEASUREMENT_FILES: ["SS_measurements.tsv"],
        OBSERVABLE_FILES: ["SS_observables.tsv"]
    }]
}
petab.write_yaml(yaml_config, "SS_conversion_reaction.yaml")

# validate written PEtab files
problem = petab.Problem.from_yaml("SS_conversion_reaction.yaml")
petab.lint_problem(problem)

measurement_df = pd.DataFrame(data={
    OBSERVABLE_ID: ['obs_b'] * nt,
    SIMULATION_CONDITION_ID: ['c0'] * nt,
    TIME: times,
    MEASUREMENT: measurements
})

observable_df = pd.DataFrame(data={
    OBSERVABLE_ID: ['obs_b'],
    OBSERVABLE_FORMULA: ['B'],
    NOISE_FORMULA: [''],
    OBSERVABLE_TRANSFORMATION: [LOG]
}).set_index([OBSERVABLE_ID])

parameter_df = pd.DataFrame(data={
    PARAMETER_ID: ['k1', 'k2'],
    PARAMETER_SCALE: [LOG] * 2,
    LOWER_BOUND: [0.0067, 0.0067],
    UPPER_BOUND: [148.413, 148.413],
    NOMINAL_VALUE: [k1, k2],
    ESTIMATE: [1, 1],
}).set_index(PARAMETER_ID)

petab.write_condition_df(condition_df, "HS_conditions.tsv")
petab.write_measurement_df(measurement_df, "HS_measurements.tsv")
petab.write_observable_df(observable_df, "HS_observables.tsv")
petab.write_parameter_df(parameter_df, "HS_parameters.tsv")

yaml_config = {
    FORMAT_VERSION: 1,
    PARAMETER_FILE: "HS_parameters.tsv",
    PROBLEMS: [{
        SBML_FILES: ["model_conversion_reaction.xml"],
        CONDITION_FILES: ["HS_conditions.tsv"],
        MEASUREMENT_FILES: ["HS_measurements.tsv"],
        OBSERVABLE_FILES: ["HS_observables.tsv"]
    }]
}
petab.write_yaml(yaml_config, "HS_conversion_reaction.yaml")

# validate written PEtab files
problem = petab.Problem.from_yaml("HS_conversion_reaction.yaml")
petab.lint_problem(problem)
