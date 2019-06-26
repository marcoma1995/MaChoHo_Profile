"""Functions for the predict step of a square root unscented Kalman filter.

The functions use pandas for most of the calculations. This means that for
most operations the order of columns or the index is irrelevant. Nevertheless,
the order might be relevant whenever matrix factorizations are involved!

"""

import pandas as pd
import numpy as np


def square_root_unscented_predict(state, root_cov, params, shock_sds, kappa):
    """Predict *state* in next period and adjust *root_cov*.
    We did not fix the mistake. 

    Args:
        state (pd.Series): period t estimate of the unobserved state vector
        root_cov (pd.DataFrame): lower triangular matrix square-root of the
            covariance matrix of the state vector in period t
        params (dict): keys are the names of the states (latent
            factors), values are series with parameters for the transition
            equation of that state.
        shock_sds (pd.Series): standard deviations of the shocks
        kappa (float): scaling parameter for the unscented predict

    Returns:
        predicted_state (pd.Series)
        predicted_root_cov (pd.DataFrame)

    """

    points = _calculate_sigma_points(state, root_cov, kappa)
    weights = _calculate_sigma_weights(state, kappa)
    transformed = _transform_sigma_points(points, params)
    predicted_state = _predict_state(transformed, weights)
    predicted_root_cov = _predict_root_cov(
        transformed, weights, predicted_state, shock_sds
    )
    return predicted_state, predicted_root_cov


def _calculate_sigma_points(state, root_cov, kappa):
    n = len(state)
    scale = np.sqrt(n + kappa)
    sigma_points = pd.concat([
        state.to_frame().T,
        state + scale * root_cov,
        state - scale * root_cov,
    ])
    sigma_points.index = range(2 * n + 1)  #Adjusted.
    return sigma_points


def _calculate_sigma_weights(state, kappa):
    n = len(state)
    first_weight = kappa / (n + kappa)
    other_weights = 1 / (2 * (n + kappa)) # Found an error here after testing.
    weight_list = [first_weight] + [other_weights] * 2 * n
    sigma_weights = pd.Series(data=weight_list, index=range(2 * n + 1))

    return sigma_weights


def _transform_sigma_points(sigma_points, params):
    factors = sigma_points.columns
    to_concat = ()
    for factor in factors:
        transformed = _cobb_douglas(sigma_points, **params[factor])
        to_concat += (transformed.rename(factor),)  #adjusted this
    out = pd.concat(to_concat, axis=1)
    return out

def _cobb_douglas(sigma_points, gammas, a):
    return a *(
            sigma_points ** gammas).product(axis=1) # Found error here.


def _predict_state(transformed_sigma_points, sigma_weights):
    return np.transpose(transformed_sigma_points).dot(sigma_weights) #Adjusted.


def _predict_root_cov(
    transformed_sigma_points,
    sigma_weights,
    predicted_state, #Found error here.
    shock_sds
    ):

    sqrt_weights = sigma_weights.apply(np.sqrt)
    deviations = transformed_sigma_points - predicted_state
    weighted_deviations = deviations.multiply(sqrt_weights, axis=0)
    # Sorting is important here.
    factors = transformed_sigma_points.columns
    shocks_root_cov = pd.DataFrame(
        data=np.diag(shock_sds[factors]), columns=factors, index=factors)
    helper_matrix = pd.concat(
        [weighted_deviations,
         shocks_root_cov])
    predicted_cov = pd.DataFrame(
        data=np.linalg.qr(helper_matrix, mode='r').T, #Found error here.
        columns=factors,
        index=factors,
    )
    return predicted_cov



