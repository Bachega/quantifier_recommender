import pandas as pd
import numpy as np
import time
import pdb


def predictQuantifierSchumacherGithub(qnt, X_test):
    """This function predict the class distribution from a given test set.
 
    Parameters
    ----------
    qnt : object
        A quantifier previously fitted from some training data.
    X_test : DataFrame
        A DataFrame of the test data.
    Returns
    -------
    array
        the class distribution of the test calculated according to the qntMethod quantifier. 
    """

    start = time.time()
    re = qnt.predict(*[np.asarray(X_test)])
    stop = time.time()
    #return stop - start
    return  re[0]