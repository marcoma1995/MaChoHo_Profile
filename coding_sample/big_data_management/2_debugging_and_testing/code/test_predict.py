import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from predict import square_root_unscented_predict
from predict import _calculate_sigma_points
from predict import _calculate_sigma_weights
from predict import _transform_sigma_points
from predict import _cobb_douglas
from predict import _predict_state
from predict import _predict_root_cov
FACTORS = list('cni')


@pytest.fixture
def setup_predict():

    out = {}
    out['state'] = pd.Series(data=[12, 10, 8], index=FACTORS)
    out['root_cov'] = pd.DataFrame(
        data=[[6, 0, 0], [3, 5, 0], [2, 1, 4.0]],
        columns=FACTORS,
        index=FACTORS,
    )
    params = {
        'c': {'gammas': pd.Series(data=[0.5] * 3, index=FACTORS), 'a': 0.5},
        'n': {'gammas': pd.Series(data=[1.5, 1, 0], index=FACTORS), 'a': 0.1},
        'i': {'gammas': pd.Series(data=[0, 0, 1.0], index=FACTORS), 'a': 1.2},
    }
    out['params'] = params
    out['shock_sds'] = pd.Series(data=[1, 2, 3.0], index=FACTORS)
    out['kappa'] = 1

    return out


@pytest.fixture
def expected_predict():
    out = {}
    out['mean'] = pd.Series(data=[13.42979972, 56.04386809, 9.6], index=FACTORS)
    out['cov'] = pd.DataFrame(
        data=[
            [126.640480, 508.664745, 33.255376],
            [508.664745, 2717.324849, 70.437680],
            [33.255376, 70.437680, 32.040000],
        ],
        columns=FACTORS,
        index=FACTORS
    )
    return out

#Test the funtions contained in square_root_unscented_predict.
    
#Test _calculate_sigma_points.
@pytest.fixture
def setup_points():
    
    out = {}
    out['state'] = pd.Series(data=[12, 10, 8], index=FACTORS)
    out['root_cov'] = pd.DataFrame(
        data=[[6, 0, 0], [3, 5, 0], [2, 1, 4.0]],
        columns=FACTORS,
        index=FACTORS,
    )
    out['kappa'] = 1
    return out

@pytest.fixture
def expected_sigma_points():
    out = {}
    out['points'] = pd.DataFrame(
            data=[[12, 10, 8],
                  [24, 10, 8],
                  [18, 20, 8],
                  [16, 12, 16],
                  [0, 10, 8],
                  [6, 0, 8],
                  [8, 8, 0]],
                  columns=FACTORS,
                  index=range(7)
    )
    return out

def test_calculate_sigma_points(setup_points, expected_sigma_points):
    calc_points = _calculate_sigma_points(**setup_points)
    assert_frame_equal(
            calc_points, expected_sigma_points['points'],check_dtype=False)

#Test _calculate_sigma_weights.
@pytest.fixture
def setup_weights():
    out = {}
    out['state']= pd.Series(data=[12, 10, 8],index=FACTORS)
    out['kappa']= 1
    return out

@pytest.fixture
def expected_sigma_weights():
    out = {}
    expected_list = [1/4]+6*[1/8]
    out['weights'] = pd.Series(data=expected_list, index=range(7))
    return out

#def test_sigma_weights_typical_input(setup_weights):
 #   expected_list=[1/4]+6*[1/8]
  #  expected=pd.Series(data=expected_list, index=range(7))
   # assert_series_equal(_calculate_sigma_weights(**setup_weights),expected)
    
def test_calculate_sigma_weigts(setup_weights, expected_sigma_weights):
    calc_weights = _calculate_sigma_weights(**setup_weights)
    assert_series_equal(calc_weights, expected_sigma_weights['weights']) 
    
"""The test above shows us an assertion error in those weights created by the 
step 'other_weights' in the function, which allowed us to spot the error in
line 57.
"""  

#Test _cobb_douglas.
@pytest.fixture
def setup_cd():
    out ={}
    out['sigma_points']=pd.DataFrame(
            data=[[12, 10, 8],
                  [24, 10, 8],
                  [18, 20, 8],
                  [16, 12, 16],
                  [0, 10, 8],
                  [6, 0, 8],
                  [8, 8, 0]],
                  columns=FACTORS,
                  index=range(7)
    )
#    params = {
 #     'c': {'gammas': pd.Series(data=[0.5] * 3, index=FACTORS), 'a': 0.5}
  #    }
    out['gammas'] = [0.5,0.5,0.5]
    out['a']=0.5
    return out

@pytest.fixture
def expected_cd():
    out = {}
    out['expected_cd'] = pd.Series([15.491933,
                   21.908902, 
                   26.832816, 
                   27.712813, 
                   0.0,
                   0.0,
                   0.0,],
                   index=range(7)
                   )
                
    return out

def test_typical_cobb_douglas(setup_cd,expected_cd):
    calc_cd= _cobb_douglas(**setup_cd)
    assert_series_equal(calc_cd, expected_cd['expected_cd'])
    
def test_input_zero_cd(setup_cd,expected_cd):
    setup_cd['sigma_points'] = pd.DataFrame(
            data=[[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]],
                  columns=FACTORS,
                  index=range(7)
    )
    expected_cd['expected_cd']= pd.Series([0.0,
                   0.0, 
                   0.0, 
                   0.0, 
                   0.0,
                   0.0,
                   0.0,],
                   index=range(7)
                   )
    calc_cd= _cobb_douglas(**setup_cd)
    assert_series_equal(calc_cd, expected_cd['expected_cd'])
    
def test_gammas_zero_cd(setup_cd,expected_cd):
    setup_cd['gammas'] = [0.0,0.0,0.0]
    expected_cd['expected_cd']= pd.Series([0.5,
                   0.5, 
                   0.5, 
                   0.5, 
                   0.5,
                   0.5,
                   0.5,],
                   index=range(7)
                   )
    calc_cd= _cobb_douglas(**setup_cd)
    assert_series_equal(calc_cd, expected_cd['expected_cd'])

"""The test above allowed us to identify the error in the Cobb-Douglas
function, because it yielded 0 instead of the expected 0.5, which was caused 
by the gammas being a factor instead of exponents.
"""         

#Test _transform_sigma_points.
@pytest.fixture
def setup_transformed_points():
    out = {}
    params = {
        'c': {'gammas': pd.Series(data=[0.5] * 3, index=FACTORS), 'a': 0.5},
        'n': {'gammas': pd.Series(data=[1.5, 1, 0], index=FACTORS), 'a': 0.1},
        'i': {'gammas': pd.Series(data=[0, 0, 1.0], index=FACTORS), 'a': 1.2},
    }
    out['params'] = params
    sigma_points = pd.DataFrame(
            data=[[12, 10, 8],
                  [24, 10, 8],
                  [18, 20, 8],
                  [16, 12, 16],
                  [0, 10, 8],
                  [6, 0, 8],
                  [8, 8, 0]],
                  columns=FACTORS,
                  index=range(7))
    out['sigma_points'] = sigma_points
    return out
    
@pytest.fixture
def expected_transformed_points():
    out = {}
    out['transformed_points'] = pd.DataFrame(
            data=[[15.491933, 41.569219, 9.600000],
                  [21.908902, 117.575508, 9.600000],
                  [26.832816, 152.735065, 9.600000],
                  [27.712813, 76.800000, 19.200000],
                  [0.0, 0.0, 9.600000],
                  [0.0, 0.0, 9.600000],
                  [0.0, 18.101934, 0.0]],
                  columns=FACTORS,
                  index=range(7))
    return out

def test_calculate_transformed_sigma_points(setup_transformed_points,
                                            expected_transformed_points):
    calc_transformed_points = _transform_sigma_points(
            **setup_transformed_points)
    assert_frame_equal(calc_transformed_points,
                       expected_transformed_points['transformed_points'],
                       check_dtype=False)
    
#Test predict state. 
@pytest.fixture 
def setup_predict_state():
    out = {}
    out['transformed_sigma_points'] = [1,2,3]
    out['sigma_weights']=[1,2,3]
    return out

@pytest.fixture
def expected_predict_state():
    out = {}
    out['result']= 14
    return out


def test_predict_state(setup_predict_state, expected_predict_state):
    calc_state = _predict_state(**setup_predict_state)
    assert(calc_state, expected_predict_state['result'])

#Test _predict_root_cov.
@pytest.fixture
def setup_predict_root_cov():
    out = {}
    out['transformed_sigma_points'] = pd.DataFrame(
        data=[[1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0]],
              columns=FACTORS,
              index=range(7))
    out['sigma_weights'] = pd.Series(data=[1/4]+6*[1/8], index=range(7))
    out['shock_sds'] = pd.Series(data=[1, 2, 3.0], index=FACTORS)
    out['predicted_state'] = [1.0,1.0,1.0]
    return out

#Wrong expected output, still looking for a good example!
@pytest.fixture
def expected_predict_root_cov():
    out ={}
    out['predict_root_cov']=pd.DataFrame(
            [        
            [-1,0.0,0.0],
            [0,-2,0.0],
            [0,0,-3]],
            columns = FACTORS,
            index = FACTORS)
    return out


def test_predict_root_cov(setup_predict_root_cov,expected_predict_root_cov):
    calc_root_cov = _predict_root_cov(**setup_predict_root_cov)
    assert_frame_equal(calc_root_cov,expected_predict_root_cov[
            'predict_root_cov'],
            check_dtype=False)
    
def test_lower_triangularity_(setup_predict_root_cov,
                                expected_predict_root_cov):
    calc_upper_right = (_predict_root_cov(
            **setup_predict_root_cov)).at['i','c']
    calc_upper_center = (_predict_root_cov(
            **setup_predict_root_cov)).at['n','i']
    calc_middle_right = (_predict_root_cov(
            **setup_predict_root_cov)).at['i','n']
    calc_results=pd.Series([calc_upper_right,
                   calc_upper_center,
                   calc_middle_right
                   ])
    expected_results = pd.Series([0,0,0])
    assert_series_equal(calc_results, expected_results,check_dtype=False) 

"""As described in the pdf, the output needs to be a lower triangular matrix. 
This test allowed us to find out, that a transpose in predict.py was missing 
and an upper triangular matrix was returned instead.
However, after finding this error, the functions still doesn't return the 
output we expect, so we decide to decompose the parts of the function to test
them one by one.
"""
def test_sqrt_weights(setup_predict_root_cov,
                                expected_predict_root_cov):
    setup_predict_root_cov['sigma_weights'] = pd.Series(
            data=[1/4]+6*[1/9], index = range(7))
    sqrt_weights = setup_predict_root_cov['sigma_weights'].apply(np.sqrt)
    assert_series_equal(sqrt_weights,pd.Series(data = [1/2]+6*[1/3],
                                               index = range(7)),
                                               check_dtype = False)
    
def test_deviations(setup_predict_root_cov,expected_predict_root_cov): 
    setup_predict_root_cov['transformed_sigma_points'] = pd.DataFrame(
        data=[[1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0]],
              columns=FACTORS,
              index=range(7))
    setup_predict_root_cov['predicted_state'] = [1,1,1]
    deviations = (setup_predict_root_cov['transformed_sigma_points']
    - setup_predict_root_cov['predicted_state'])
    assert_frame_equal(deviations,pd.DataFrame(
        data=[[0.0,0.0,0.0],
              [0.0,0.0,0.0],
              [0.0,0.0,0.0],
              [0.0,0.0,0.0],
              [0.0,0.0,0.0],
              [0.0,0.0,0.0],
              [0.0,0.0,0.0]],
              columns=FACTORS,
              index=range(7)) )

def test_weighted_deviations(setup_predict_root_cov,expected_predict_root_cov):
    setup_predict_root_cov['transformed_sigma_points'] = pd.DataFrame(
        data=[[1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0]],
              columns=FACTORS,
              index=range(7))
    setup_predict_root_cov['sigma_weights'] = pd.Series(
            data=[1/4]+6*[1/9], index = range(7))
    setup_predict_root_cov['predicted_state'] = [0,0,0]
    sqrt_weights = setup_predict_root_cov['sigma_weights'].apply(np.sqrt)
    deviations = (setup_predict_root_cov['transformed_sigma_points']
    - setup_predict_root_cov['predicted_state'])
    weighted_deviations = deviations.multiply(sqrt_weights, axis=0)
    assert_frame_equal(weighted_deviations, pd.DataFrame(
            data=[[0.5,0.5,0.5],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3]],
                  columns=FACTORS,
                  index=range(7)))
    
def test_shock_root_cov(setup_predict_root_cov,expected_predict_root_cov): 
    setup_predict_root_cov['shock_sds'] = pd.Series(
            data=[1, 2, 3.0], index=FACTORS)
    factors = setup_predict_root_cov['transformed_sigma_points'].columns    
    shocks_root_cov = pd.DataFrame(
        data=np.diag(setup_predict_root_cov['shock_sds'][factors]),
        columns=factors, index=factors)
    assert_frame_equal(shocks_root_cov,pd.DataFrame([[1,0,0],
                                                     [0,2,0],
                                                     [0,0,3]],
                                                    index = factors,
                                                    columns = factors),
                                                    check_dtype =False)

def test_helper_matrix(setup_predict_root_cov,expected_predict_root_cov):
    setup_predict_root_cov['transformed_sigma_points'] = pd.DataFrame(
        data=[[1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0]],
              columns=FACTORS,
              index=range(7))
    setup_predict_root_cov['sigma_weights'] = pd.Series(
            data=[1/4]+6*[1/9], index = range(7))
    setup_predict_root_cov['predicted_state'] = [0,0,0]
    setup_predict_root_cov['shock_sds'] = pd.Series(
            data=[1, 2, 3.0], index=FACTORS)
    sqrt_weights = setup_predict_root_cov['sigma_weights'].apply(np.sqrt)
    deviations = (setup_predict_root_cov['transformed_sigma_points']
    - setup_predict_root_cov['predicted_state'])
    weighted_deviations = deviations.multiply(sqrt_weights, axis=0)
    factors = setup_predict_root_cov['transformed_sigma_points'].columns    
    shocks_root_cov = pd.DataFrame(
        data=np.diag(setup_predict_root_cov['shock_sds'][factors]),
        columns=factors, index=factors)
    helper_matrix = pd.concat(
        [weighted_deviations,
         shocks_root_cov])
    assert_frame_equal(helper_matrix, pd.DataFrame(
            data=[[0.5,0.5,0.5],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1,0,0],
                  [0,2,0],
                  [0,0,3]],                   
                  columns=FACTORS,
                  index=[0, 1, 2, 3, 4, 5, 6, 'c', 'n', 'i']),
    check_dtype =False)

def test_predict_cov(setup_predict_root_cov,expected_predict_root_cov):
    helper_matrix = pd.DataFrame(
            data=[[0.5,0.5,0.5],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1,0,0],
                  [0,2,0],
                  [0,0,3]],                   
                  columns=FACTORS,
                  index=[0, 1, 2, 3, 4, 5, 6, 'c', 'n', 'i'])
    factors = ['c','n','i']
    predicted_cov = pd.DataFrame(
        data=np.linalg.qr(helper_matrix, mode='r').T,
        columns=factors,
        index=factors,
    )
    assert_frame_equal(predicted_cov,pd.DataFrame(
            [
            [-1.38443731,0,0],
            [-0.66212219,-2.11619018,0],
            [-0.66212219,-0.22600089,-3.07037204]
            ],
            columns = factors,
            index = factors
            ))
"""As all those tests testing the steps within the function pass, we had 
to look for some other issue which might cause wrong output. We thereby came 
up with checking the order of the input called in the function and found that 
switching shock_sds and predict_state solves the error.
"""    
    
    
def test_square_root_unscented_predict_mean(setup_predict, expected_predict):
    calc_mean, calc_root_cov = square_root_unscented_predict(**setup_predict)
    assert_series_equal(calc_mean, expected_predict['mean'])


def test_square_root_unscented_predict_cov_values(setup_predict,
                                                  expected_predict):
    calc_mean, calc_root_cov = square_root_unscented_predict(**setup_predict)
    calc_cov = calc_root_cov.dot(calc_root_cov.T)
    assert_frame_equal(calc_cov, expected_predict['cov'])