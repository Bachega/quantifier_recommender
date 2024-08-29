import pandas as pd
import numpy as np
from utils.helpers import load_class
import time
import pdb


def fitQuantifierSchumacherGithub(qntMethod, X_train, y_train):
    """This function fit a quantifier using the codes provided by Tobias Schumacher.
 
    Parameters
    ----------
    qntMethod : string
        Quantification method name, according to the alg_index.csv file
    X_train : DataFrame
        A DataFrame of the training data.
    y_train : DataFrame
        A DataFrame with the training labels.
    Returns
    -------
    object
        the quantifier fitted. 
    """

    algorithm_index = pd.read_csv("./utils/alg_index.csv",
                                sep=";",
                                index_col="algorithm")

    algorithm_index = algorithm_index.loc[algorithm_index.export == 1]
    algorithms = list(algorithm_index.index)

    algorithm_dict = dict({alg: load_class(algorithm_index.loc[alg, "module_name"],
                                                algorithm_index.loc[alg, "class_name"])
                        for alg in algorithms})

    init_args = []

    fit_args = [np.asarray(X_train), np.asarray(y_train)]   
    qf = algorithm_dict[qntMethod](*init_args)

    qf.fit(*fit_args)

    return qf  
