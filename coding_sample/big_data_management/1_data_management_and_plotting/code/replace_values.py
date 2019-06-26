# Import packages.
import numpy as np

# Create functions.

# Create a function which sets values of -100 to np.nan.
def neg100_to_missing(input):
    """ Set the input to np.nan if its value is -100.
    
    """
    if(input == -100):
        input = np.nan
    return input

# Define a function that sets negative values to np.nan.
def neg_to_missing(input):
    """Set the input to np.nan if its value is below 0.
    
    """
    if(input < 0):
        input = np.nan
    return input

# Define a function that sets values of 1 to 0 and vice versa.
def invert_bool(x):
    """Invert a boolean input of 1 to 0 and 0 to 1.
    
    """
    if x == 0:
        return 1
    if x ==1:
        return 0