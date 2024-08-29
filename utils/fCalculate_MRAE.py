import numpy as np

def fCalculateRAE(absolute_error, true_value):
    if(true_value != 0):
        relative_error = absolute_error/true_value
    else:
        relative_error = absolute_error

    return np.round(relative_error, 2)