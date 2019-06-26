import numpy as np

import pytest

from update import fast_batch_update

# Define the data used in the tests.


@pytest.fixture
def setup_fast_batch_update_normal_input_and_small_covariance():
    out = {}
    out["states"] = np.array([[1, 2], [2, 2]])
    out["root_covs"] = np.array([[[1, 0], [1, 1]],
                                [[0.001, 0], [0.002, 0.003]]])
    out["measurements"] = np.array([[2.0, 3.0]])
    out["loadings"] = np.array([1, 1])
    out["meas_var"] = 1.0
    return out


@pytest.fixture
def expected_fast_batch_normal_input_and_small_covariance():
    out = {}
    out["expected_states"] = np.array([[2 / 3, 3 / 2], [1.999997, 1.999985]])
    out["expected_root_covs"] = np.array(
        [
            [[-5.773503e-01, 0.000000], [2.775558e-16, 0.707107]],
            [[-0.001, 0.000], [-0.002, 0.003]],
        ]
    )
    return out


@pytest.fixture
def setup_fast_batch_update_large_states():
    out = {}
    out["states"] = np.array([[1, 2], [100, 200]])
    out["root_covs"] = np.array([[[1, 0], [1, 1]], [[10, 0], [10, 10]]])
    out["measurements"] = np.array([2.0, 900.0])
    out["loadings"] = np.array([1, 1])
    out["meas_var"] = 20.0
    return out


@pytest.fixture
def expected_fast_batch_update_large_states():
    out = {}
    out["expected_states"] = np.array([[0.92, 1.88], [330.769, 546.154]])
    out["expected_root_covs"] = np.array(
        [[[-0.917, 0.000000], [-0.829, 0.976]],
         [[-4.804, 0.000000], [3.203, 4.082]]]
    )
    return out


@pytest.fixture
def setup_error_pandas_fastbatch():
    out = {}
    out["states"] = np.array([[1, 2], [2, 2]])
    out["root_covs"] = np.array([[[1, 0], [1, 1]],
                                [[0.001, 0], [0.002, 0.003]]])
    out["measurements"] = np.array([2.0, 3.0])
    out["loadings"] = np.array([1, 1])
    out["meas_var"] = 1.0
    return out


# Test if the correct output is being computed if the input is of correct type
# and valid.


def test_fast_batch_states_normal_input_and_small_covariance(
    setup_fast_batch_update_normal_input_and_small_covariance,
    expected_fast_batch_normal_input_and_small_covariance,
):
    calc_fast_batch_states, calc_root_covs = fast_batch_update(
        **setup_fast_batch_update_normal_input_and_small_covariance
    )
    np.testing.assert_array_almost_equal(
        expected_fast_batch_normal_input_and_small_covariance[(
            "expected_states")],
        calc_fast_batch_states,
        decimal=3,
    )


def test_fast_batch_root_covs_normal_input_and_small_covariance(
    setup_fast_batch_update_normal_input_and_small_covariance,
    expected_fast_batch_normal_input_and_small_covariance,
):
    calc_fast_batch_states, calc_root_covs = fast_batch_update(
        **setup_fast_batch_update_normal_input_and_small_covariance
    )
    np.testing.assert_array_almost_equal(
        expected_fast_batch_normal_input_and_small_covariance[(
            "expected_root_covs")],
        calc_root_covs,
        decimal=3,
    )


def test_fast_batch_states_large_states(
    setup_fast_batch_update_large_states,
    expected_fast_batch_update_large_states
):
    calc_fast_batch_states, calc_root_covs = fast_batch_update(
        **setup_fast_batch_update_large_states
    )
    np.testing.assert_array_almost_equal(
        expected_fast_batch_update_large_states["expected_states"],
        calc_fast_batch_states,
        decimal=3,
    )


def test_fast_batch_root_covs_large_states(
    setup_fast_batch_update_large_states,
    expected_fast_batch_update_large_states
):
    calc_fast_batch_states, calc_root_covs = fast_batch_update(
        **setup_fast_batch_update_large_states
    )
    np.testing.assert_array_almost_equal(
        expected_fast_batch_update_large_states["expected_root_covs"],
        calc_root_covs,
        decimal=3,
    )


# Test if the correct error is raised if the input has wrong dimensions.

# Test if a ValueError is raised if root_covs is wrong dimension.
def test_fastbatch_root_covs_wrong_dim(setup_error_pandas_fastbatch):
    root_covs_too_few = np.array([[1, 0], [1, 1]])
    setup_error_pandas_fastbatch["root_covs"] = root_covs_too_few
    with pytest.raises(ValueError):
        fast_batch_update(**setup_error_pandas_fastbatch)


# Test if a ValueError is raised if too many measurements.
def test_fastbatch_measurements_too_long(setup_error_pandas_fastbatch):
    measurements_too_long = np.array([2.0, 3.0, 1.0])
    setup_error_pandas_fastbatch["measurements"] = measurements_too_long
    with pytest.raises(ValueError):
        fast_batch_update(**setup_error_pandas_fastbatch)


# Test if a ValueError is raised if loadings is wrong dimension
def test_fastbatch_dimension(setup_error_pandas_fastbatch):
    loadings_too_long = np.array([1, 1, 1])
    setup_error_pandas_fastbatch["loadings"] = loadings_too_long
    with pytest.raises(ValueError):
        fast_batch_update(**setup_error_pandas_fastbatch)
