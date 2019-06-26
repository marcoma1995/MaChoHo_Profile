import numpy as np

import pandas as pd

from time import time

from update import fast_batch_update

from update import fast_batch_update_approach_1

from update import fast_batch_update_approach_2

from update import fast_batch_update_approach_3

from update import pandas_batch_update


# Load and prepare data.
data = pd.read_stata("../chs_data.dta")
data.replace(-100, np.nan, inplace=True)
data = data.query("age == 0")
data.reset_index(inplace=True)
data = data["weightbirth"]
data.fillna(data.mean(), inplace=True)

# Fix dimensions.
nobs = len(data)
state_names = ["cog", "noncog", "mother_cog", "mother_noncog", "investments"]
nstates = len(state_names)

# Construct initial states.
states_np = np.zeros((nobs, nstates))
states_pd = pd.DataFrame(data=states_np, columns=state_names)

# Construct initial covariance matrices.
root_cov = np.linalg.cholesky(
    [
        [0.1777, -0.0204, 0.0182, 0.0050, 0.0000],
        [-0.0204, 0.2002, 0.0592, 0.0261, 0.0000],
        [0.0182, 0.0592, 0.5781, 0.0862, -0.0340],
        [0.0050, 0.0261, 0.0862, 0.0667, -0.0211],
        [0.0000, 0.0000, -0.0340, -0.0211, 0.0087],
    ]
)

root_covs_np = np.zeros((nobs, nstates, nstates))
root_covs_np[:] = root_cov

root_covs_pd = []
for i in range(nobs):
    root_covs_pd.append(
        pd.DataFrame(data=root_cov, columns=state_names, index=state_names)
    )

# Construct measurements.
meas_bwght_np = data.values
meas_bwght_pd = data

# Construct loadings.
loadings_bwght_np = np.array([1.0, 0, 0, 0, 0])
loadings_bwght_pd = pd.Series(loadings_bwght_np, index=state_names)

# Construct the variance.
meas_var_bwght = 0.8


# Time pandas_batch_upate.
runtimes = []
for i in range(2):
    start = time()
    pandas_batch_update(
        states=states_pd,
        root_covs=root_covs_pd,
        measurements=meas_bwght_pd,
        loadings=loadings_bwght_pd,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)

print("pandas_batch_update took {} seconds.".format(np.mean(runtimes)))

# Time the first speed improvment approach.
runtimes = []
for i in range(100):
    start = time()
    fast_batch_update_approach_1(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np,
        loadings=loadings_bwght_np,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)

print("fast_batch_update_approach_1 took {} seconds.".format((
    np.mean(runtimes))))

# Time the second speed improvment approach.
runtimes = []
for i in range(100):
    start = time()
    fast_batch_update_approach_2(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np,
        loadings=loadings_bwght_np,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)

print("fast_batch_update_approach_2 took {} seconds.".format((
    np.mean(runtimes))))

# Time the third speed improvment approach.
runtimes = []
for i in range(100):
    start = time()
    fast_batch_update_approach_3(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np,
        loadings=loadings_bwght_np,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)

print("fast_batch_update_approach_3 took {} seconds.".format((
    np.mean(runtimes))))

# Time the final solution.
runtimes = []
for i in range(100):
    start = time()
    fast_batch_update(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np,
        loadings=loadings_bwght_np,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)

print("fast_batch_update took {} seconds.".format(np.mean(runtimes)))
