import pandas as pd
import numpy as np

def square_root_linear_update(state,
                              root_cov,
                              measurement,
                              loadings,
                              meas_var):
    """Update *state* and *root_cov with* with a *measurement*.

    Args:
        state (pd.Series): pre-update estimate of the unobserved state vector
        root_cov (pd.DataFrame): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (pd.Series): the factor loadings
        meas_var(float):  l diagonal element of cov matrix of measment error

    Returns:
        updated_state (pd.Series)
        updated_root_cov (pd.Series)
    
    Note:
        We worked with the uncorrected version of the Kalman Filter PDF.

    """

    intermediate_result_star = _intermediate_result_star(root_cov, loadings)
    predicted_measurement = _predicted_measurement(state, loadings)
    residual = _residual(predicted_measurement, measurement)
    matrix = _matrix(root_cov, intermediate_result_star, meas_var, state)
    matrix_decomposition = _matrix_decomposition(matrix)
    kalman_gain = _kalman_gain(intermediate_result_star, matrix_decomposition)
    updated_state = _updated_state(residual, state, kalman_gain)
    updated_root_cov = _updated_root_cov(matrix_decomposition)
    
    return updated_state, updated_root_cov


def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    
def is_lower_triangular(A):
   if np.allclose(A, np.tril(A)):
       return True
   else:
       return False

#Equation 3.   
def _predicted_measurement(state, loadings):   
    
    if len(state) != len(loadings):
        raise ValueError(
                """Length of state and length of loadings need to be equal."""
                )
        
    else:
        predicted_measurement = state.dot(loadings)
        return predicted_measurement

#Equation 4.
def  _residual(predicted_measurement, measurement):  
    residual = measurement - predicted_measurement
    return residual

#Equation 16.
def  _intermediate_result_star(root_cov, loadings):  
    
    if (len(loadings) != len(
            root_cov.index)) or (len(
                            root_cov.index) != len(root_cov.columns)):
        raise ValueError(
                """root_cov needs to be a  square matrix and the length of 
                loadings needs to be the same as the number of columns of 
                root_cov."""
                )
        
    elif is_lower_triangular(root_cov) == False:
        raise TypeError("root_cov needs to be lower triangular.")
    
    elif is_pos_def(root_cov.dot(np.transpose(root_cov))) == False:
        raise TypeError(
                """root_cov multiplied by its transpose needs to be positive 
                definite in order to be a root of a covariance matrix."""
                )
        
    else:
        intermediate_result_star = np.transpose(root_cov).dot(np.transpose(
                loadings))
        return intermediate_result_star
        
#Equation 17.
def  _matrix(root_cov, intermediate_result_star, meas_var, state): 
    
    if meas_var <= 0:
        raise ValueError("meas_var needs to be strictly positive.")
       
    else:
        n = len(state)
        matrix = np.zeros((n+1,n+1))
        matrix[0,0] = np.sqrt(meas_var)
        matrix[1:,1:] = np.transpose(root_cov)
        matrix[1:,0] = intermediate_result_star
        
    return matrix

#Equation 18.
def  _matrix_decomposition(matrix): 
    matrix_decomposition = np.linalg.qr(matrix, mode='r')
    return matrix_decomposition

#Equation 7.
def _kalman_gain(intermediate_result_star, matrix_decomposition): 
    variance = np.square(matrix_decomposition[0,0])
    intermediate_result = (matrix_decomposition[0,0])*(np.transpose(
            matrix_decomposition[0,1:]))
    kalman_gain = intermediate_result.dot(1/variance)
    return kalman_gain

#Equation 8.
def _updated_state(residual, state, kalman_gain): 
    updated_state = state + kalman_gain * residual
    return updated_state

def _updated_root_cov(matrix_decomposition):
    root_cov = np.transpose(matrix_decomposition[1:,1:])
    
    updated_root_cov = pd.DataFrame(
            data=root_cov,
    )
    
    return updated_root_cov

