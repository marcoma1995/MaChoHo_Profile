import pandas as pd
import pytest

from update import square_root_linear_update
from pandas.testing import assert_frame_equal, assert_series_equal

# Define the data used in the tests.

@pytest.fixture
def setup_update():
    out= {}
    out['state']= pd.Series(data=[1, 2])
    out['root_cov']= pd.DataFrame(
        data=[[1, 0], [1, 1]],
    )
    out['measurement']= 2.0
    out['loadings']= pd.Series(data=[1, 1])
    out['meas_var']= 1
    return out
 
@pytest.fixture
def expected_update():
    out = {}
    out['state'] = pd.Series(data=[2/3, 3/2])
    out['cov']= pd.DataFrame(
            data=[[1/3,0],[0,1/2]]
    ) 
    return out

@pytest.fixture
def setup_small_covariance_matrix_update():
    out= {}
    out['state']= pd.Series(data=[1, 2])
    out['root_cov']= pd.DataFrame(
        data=[[0.001, 0], [0.002, 0.003]],
    )
    out['measurement']= 2.0
    out['loadings']= pd.Series(data=[1, 1])
    out['meas_var']= 1
    return out

@pytest.fixture
def expected_update_small_covariance_matrix():
    out = {}
    out['state'] = pd.Series(data=[0.9999970001, 1.999985])
    out['cov']= pd.DataFrame(
            data=[
                    [9.999910002e-07,1.999955001e-06],
                    [1.999955001e-06,1.2999775e-05]
                    ]
    ) 
    return out

@pytest.fixture
def setup_error_update():
    out= {}
    out['state']= test_state_2
    out['root_cov']= test_root_cov_triangular
    out['measurement']= test_measurement
    out['loadings']= test_loadings_2
    out['meas_var']= 15
    return out

@pytest.fixture
def setup_large_states_update():
    out= {}
    out['state']= pd.Series(data=[300, 500])
    out['root_cov']= pd.DataFrame(
        data=[[10, 0], [10, 10]],
    )
    out['measurement']= 900.0
    out['loadings']= pd.Series(data=[1, 1])
    out['meas_var']= 50
    return out

@pytest.fixture
def expected_update_large_states():
    out = {}
    out['state'] = pd.Series(data=[3700/11,6100/11])
    out['cov']= pd.DataFrame(
            data=[[300/11,-100/11],[-100/11,400/11]]
    ) 
    return out

cov_data_triangular = {'A':[1,26],'B':[0,34]}
test_root_cov_triangular = pd.DataFrame(cov_data_triangular)
    
cov_data_large = {'A':[1,26,4],'B':[0,34,5], 'C':[0,0,1]}
test_root_cov_large = pd.DataFrame(cov_data_large)

cov_data_not_square = {'A':[1],'B':[0]}
test_root_cov_not_square = pd.DataFrame(cov_data_not_square)

cov_data_not_triangular = {'A':[1,26],'B':[7,34]}
test_root_cov_not_triangular = pd.DataFrame(cov_data_not_triangular)

cov_data_zero = {'A':[0,0],'B':[0,0]}
test_root_cov_zero = pd.DataFrame(cov_data_zero)

data_state_2 = [1,2]
test_state_2 = pd.Series(data_state_2)

data_state_3 = [1,2,3]
test_state_3 = pd.Series(data_state_3)

data_loadings_2 = [0.4, 0.6]
test_loadings_2 = pd.Series(data_loadings_2)

data_loadings_3 = [0.4, 0.6, 0.2]
test_loadings_3 = pd.Series(data_loadings_3)

test_measurement = 4

# Test if the correct output is being computed if the input is of correct type
# and valid.

# Test it with simple output.
def test_square_root_linear_update_state_normal_input(
        setup_update,
        expected_update
        ):
    calc_state, calc_root_cov = square_root_linear_update(**setup_update)
    assert_series_equal(calc_state, expected_update['state'])
    
def test_square_root_linear_update_root_cov_normal_input(
        setup_update, 
        expected_update
        ):
    calc_state, calc_root_cov = square_root_linear_update(**setup_update)
    calc_cov = calc_root_cov.dot(calc_root_cov.T)
    assert_frame_equal(calc_cov, expected_update['cov'])
    
# Test it with a root covariance matrix with very small entries.        
def test_square_root_linear_update_state_small_covariance(
        setup_small_covariance_matrix_update, 
        expected_update_small_covariance_matrix
        ):
    calc_state, calc_root_cov = square_root_linear_update(
            **setup_small_covariance_matrix_update
            )
    assert_series_equal(
            calc_state,
            expected_update_small_covariance_matrix['state']
            )
 
def test_square_root_linear_update_root_cov_small_covariance(
        setup_small_covariance_matrix_update,
        expected_update_small_covariance_matrix
        ):
    calc_state, calc_root_cov = square_root_linear_update(
            **setup_small_covariance_matrix_update
            )
    calc_cov = calc_root_cov.dot(calc_root_cov.T)
    assert_frame_equal(
            calc_cov, 
            expected_update_small_covariance_matrix['cov']
            )
# Test it with large states as inputs.
def test_square_root_linear_update_state_large_states(
        setup_large_states_update,
        expected_update_large_states
        ):
    calc_state, calc_root_cov = square_root_linear_update(
            **setup_large_states_update
            )
    assert_series_equal(calc_state, expected_update_large_states['state'])
    
def test_square_root_linear_update_root_cov_large_states(
        setup_large_states_update, 
        expected_update_large_states
        ):
    calc_state, calc_root_cov = square_root_linear_update(
            **setup_large_states_update
            )
    calc_cov = calc_root_cov.dot(calc_root_cov.T)
    assert_frame_equal(calc_cov, expected_update_large_states['cov'])  
    
# Test if the correct error is raised if the input has wrong dimensions.
    
# If state-vector is too long test if a ValueError occurs.
def test_update_state_vector_too_long(setup_error_update):
    data_state_3 = [1,2,3]
    test_state_3 = pd.Series(data_state_3)
    setup_error_update['state'] = test_state_3
    with pytest.raises(ValueError):
        square_root_linear_update(**setup_error_update)

# If loadings-vector is too long test if a ValueError occurs.
def test_update_loadings_vector_too_long(setup_error_update):
    setup_error_update['loadings'] = test_loadings_3
    with pytest.raises(ValueError):
        square_root_linear_update(**setup_error_update)

# If covariance-matrix is too large test if ValueError occurs.
def test_update_cov_matrix_too_large(setup_error_update):
    setup_error_update['root_cov'] = test_root_cov_large
    with pytest.raises(ValueError):
        square_root_linear_update(**setup_error_update)

# If covariance-matrix is not a square-matrix test if ValueError occurs.
def test_update_cov_matrix_not_square(setup_error_update):
    setup_error_update['state'] = test_state_3
    setup_error_update['root_cov'] = test_root_cov_not_square
    with pytest.raises(ValueError):
        square_root_linear_update(**setup_error_update)


# Test if the correct exception is raised if nonsensial input is used.

# Test if a ValueError is raised if meas_var equals 0.
def test_update_meas_var_equals_zero(setup_error_update):
    setup_error_update['meas_var'] = 0
    with pytest.raises(ValueError):
        square_root_linear_update(**setup_error_update)

# Test if a ValueError is raised if meas_var is negative.
def test_update__meas_var_negative(setup_error_update):
    setup_error_update['meas_var'] = -1
    with pytest.raises(ValueError):
        square_root_linear_update(**setup_error_update)

# Test if a ValueError is raised if the covariance matrix is not triangular.
def test_update_cov_matrix_not_lower_triangular(setup_error_update):
    setup_error_update['root_cov'] = test_root_cov_not_triangular
    setup_error_update['meas_var'] = 1
    with pytest.raises(TypeError):
        square_root_linear_update(**setup_error_update)

# Test if a TypeError is raised if the covariance matrix only consists of zeros
# hence the matrix times its transpose is not positive definite.
def test_update_cov_matrix_zero(setup_error_update):
    setup_error_update['root_cov'] = test_root_cov_zero
    setup_error_update['meas_var'] = 1
    with pytest.raises(TypeError):
        square_root_linear_update(**setup_error_update)
