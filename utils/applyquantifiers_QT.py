from QT import QT
from QT_ACC import QT_ACC

"""This function is an interface for running different quantification methods.
 
Parameters
----------
qntMethod : string
    Quantification method name
test : array
    A numeric vector of scores predicted from the test set.
dt_name : string
    name of dataset input by the user.
folder : string
    path of folder to save the results.

Returns
-------
array
    the class distribution of the test calculated according to the qntMethod quantifier. 
 """
def apply_quantifier_QT(qntMethod, test, dt_name, folder):

    if qntMethod == "QT":
        return QT(test,dt_name,folder )
    if qntMethod == "QT_ACC":
        return QT_ACC(test,dt_name,folder)
    
