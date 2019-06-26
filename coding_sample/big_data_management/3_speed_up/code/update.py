import numba

import numpy as np

import pandas as pd


def pandas_update(state, root_cov, measurement, loadings, meas_var):
    """Update *state* and *root_cov* with with a *measurement*.

    Args:
        state (pd.Series): pre-update estimate of the unobserved state vector
        root_cov (pd.DataFrame): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (pd.Series): the factor loadings
        meas_var(float): variance of the measurement error
    Returns:
        updated_state (pd.Series)
        updated_root_cov (pd.DataFrame)

    """
    expected_measurement = state.dot(loadings)
    residual = measurement - expected_measurement
    f_star = root_cov.T.dot(loadings)
    first_row = pd.DataFrame(
        data=[np.sqrt(meas_var)] + [0] * len(loadings),
        index=[0] + list(state.index)
    ).T
    other_rows = pd.concat([f_star, root_cov.T], axis=1)
    m = pd.concat([first_row, other_rows])
    r = np.linalg.qr(m, mode="r")
    root_sigma = r[0, 0]
    kalman_gain = pd.Series(r[0, 1:], index=state.index) / root_sigma
    updated_root_cov = pd.DataFrame(
        data=r[1:, 1:], columns=state.index, index=state.index
    ).T
    updated_state = state + kalman_gain * residual

    return updated_state, updated_root_cov


def pandas_batch_update(states, root_covs, measurements, loadings, meas_var):
    """Call pandas_update repeatedly.

    Args:
        states (pd.DataFrame)
        root_covs (list)
        measurements (pd.Series)
        loadings (pd.Series)
        meas_var (float)
    Returns:
        updated_states (pd.DataFrame)
        updated_root_covs (list)

    """
    out_states = []
    out_root_covs = []
    for i in range(len(states)):
        updated_state, updated_root_cov = pandas_update(
            state=states.loc[i],
            root_cov=root_covs[i],
            measurement=measurements[i],
            loadings=loadings,
            meas_var=meas_var,
        )
        out_states.append(updated_state)
        out_root_covs.append(updated_root_cov)
    out_states = pd.concat(out_states, axis=1).T
    return out_states, out_root_covs


def fast_batch_update(states, root_covs, measurements, loadings, meas_var):
    """Update state estimates for a whole dataset.

    Let nstates be the number of states and nobs the number of observations.
    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):

    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates,
        nstates)
    """
    n_obs = states.shape[0]
    shape_of_states = np.shape(states)
    out_states = np.zeros(shape_of_states)
    out_root_covs = np.zeros((n_obs, shape_of_states[1], shape_of_states[1]))
    root_covs_transpose = root_covs.transpose(0, 2, 1)
    expected_measurement = np.linalg.multi_dot([states, loadings])
    f_star = np.linalg.multi_dot([root_covs_transpose,
                                 loadings])[:,
                                            np.newaxis].reshape(n_obs, -1, 1)
    residual = np.reshape(np.subtract(measurements, expected_measurement),
                          (-1, 1))
    first_row_3d = np.ones(shape=(len(states), 1, len(loadings) + 1))\
        * np.append([np.sqrt(meas_var)], np.zeros(len(loadings)))
    other_rows = np.concatenate((f_star, root_covs_transpose), axis=2)
    m = np.concatenate((first_row_3d, other_rows), axis=1)
    r = qr_decomposition(m, n_obs)
    root_sigma = r[:, 0, 0].reshape(-1, 1)
    kalman_gain = np.divide(r[:, 0, 1:], root_sigma)
    out_states = states + np.multiply(residual, kalman_gain)
    out_root_covs = r[:, 1:, 1:].transpose(0, 2, 1)
    return out_states, out_root_covs


@numba.jit(nopython=True)
def qr_decomposition(m, n_obs):
    r = np.empty(m.shape)
    for i in range(n_obs):
        q, r_help = np.linalg.qr(m[i])
        r[i] = r_help
    return r


def fast_batch_update_approach_1(states,
                                 root_covs,
                                 measurements,
                                 loadings,
                                 meas_var):
    """Update state estimates for a whole dataset.

    Let nstates be the number of states and nobs the number of observations.

    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):

    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates,
        nstates)
    """
    shape_of_states = states.shape
    n_obs = states.shape[0]
    out_states = np.zeros(shape_of_states)
    out_root_covs = np.zeros((n_obs, shape_of_states[1], shape_of_states[1]))
    root_covs_transpose = root_covs.transpose(0, 2, 1)
    expected_measurement = np.einsum("i,ji->j", loadings, states)
    f_star = np.dot(root_covs_transpose,
                    loadings)[:, np.newaxis].reshape(n_obs, -1, 1)
    residual_help = np.subtract(measurements, expected_measurement)
    residual = residual_help.reshape(-1, 1)
    first_row = np.array([np.append([np.sqrt(meas_var)],
                         np.zeros(len(loadings)))])
    other_rows = np.concatenate((f_star, root_covs_transpose), axis=2)
    first_row_3d = np.ones(shape=(len(states), 1, len(first_row))) * first_row
    m = np.concatenate((first_row_3d, other_rows), axis=1)
    r = np.empty(np.shape(m))
    for i in range(n_obs):
        r[i] = np.linalg.qr(m[i], mode="r")
        root_sigma = r[i, 0, 0]
        kalman_gain = np.array([np.divide(r[i, 0, 1:], root_sigma)])
        out_states[i] = states[i] + np.multiply(residual[i], kalman_gain)
    out_root_covs = r[:, 1:, 1:].transpose(0, 2, 1)
    return out_states, out_root_covs


def fast_batch_update_approach_2(states,
                                 root_covs,
                                 measurements,
                                 loadings,
                                 meas_var):
    """Update state estimates for a whole dataset.

    Let nstates be the number of states and nobs the number of observations.

    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):

    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates,
        nstates)
    """
    shape_of_states = states.shape
    n_obs = states.shape[0]
    out_states = np.zeros(shape_of_states)
    out_root_covs = np.zeros((n_obs, shape_of_states[1], shape_of_states[1]))
    root_covs_transpose = root_covs.transpose(0, 2, 1)
    expected_measurement = np.einsum("i,ji->j", loadings, states)
    f_star = np.dot(root_covs_transpose,
                    loadings)[:, np.newaxis].reshape(n_obs, -1, 1)
    residual_help = np.subtract(measurements, expected_measurement)
    residual = residual_help.reshape(-1, 1)
    first_row = np.array([np.append([np.sqrt(meas_var)],
                         np.zeros(len(loadings)))])
    other_rows = np.concatenate((f_star, root_covs_transpose), axis=2)
    first_row_3d = np.ones(shape=(len(states), 1, len(first_row))) * first_row
    m = np.concatenate((first_row_3d, other_rows), axis=1)
    r = np.array([np.linalg.qr(m[i], mode="r") for i in range((n_obs))])
    root_sigma_help = r[:, 0, 0]
    root_sigma = root_sigma_help.reshape(-1, 1)
    kalman_gain = np.divide(r[:, 0, 1:], root_sigma)
    out_states = states + np.multiply(residual, kalman_gain)
    out_root_covs = r[:, 1:, 1:].transpose(0, 2, 1)
    return out_states, out_root_covs


def fast_batch_update_approach_3(states,
                                 root_covs,
                                 measurements,
                                 loadings,
                                 meas_var):
    """Update state estimates for a whole dataset.

    Let nstates be the number of states and nobs the number of observations.

    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):

    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates,
        nstates)
    """
    shape_of_states = states.shape
    n_obs = states.shape[0]
    out_states = np.zeros(shape_of_states)
    out_root_covs = np.zeros((n_obs, shape_of_states[1], shape_of_states[1]))
    root_covs_transpose = root_covs.transpose(0, 2, 1)
    expected_measurement = np.einsum("i,ji->j", loadings, states)
    f_star = np.dot(root_covs_transpose,
                    loadings)[:, np.newaxis].reshape(n_obs, -1, 1)
    residual_help = np.subtract(measurements, expected_measurement)
    residual = residual_help.reshape(-1, 1)
    first_row = np.array([np.append([np.sqrt(meas_var)],
                                    np.zeros(len(loadings)))])
    other_rows = np.concatenate((f_star, root_covs_transpose), axis=2)
    first_row_3d = np.ones(shape=(len(states), 1, len(first_row))) * first_row
    m = np.concatenate((first_row_3d, other_rows), axis=1)
    r = np.empty(np.shape(m))
    for i in range(n_obs):
        r[i] = np.linalg.qr(m[i], mode="r")
    root_sigma_help = r[:, 0, 0]
    root_sigma = root_sigma_help.reshape(-1, 1)
    kalman_gain = np.divide(r[:, 0, 1:], root_sigma)
    out_states = states + np.multiply(residual, kalman_gain)
    out_root_covs = r[:, 1:, 1:].transpose(0, 2, 1)
    return out_states, out_root_covs
