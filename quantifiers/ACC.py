import numpy as np
import pdb

def ACC(test_scores, tprfpr, thr = 0.5):
    """Adjust Classify & Count (ACC)

    It quantifies events based on trained classifier and correct the estimate using TPR and FPR, applying Adjust Classify & Count (ACC) method, according to Forman (2005).
    
    Parameters
    ----------
    Test scores : array
        A numeric vector of scores predicted from the test set.
    TprFpr : matrix
        A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
    thr : float  
        The threshold value for hard predictions. Default value = 0.5.
    
    Returns
    -------
    array
        the class distribution of the test. 
    """
    
    TprFpr = tprfpr[tprfpr['threshold'] == thr]
    tpr = float(TprFpr['tpr'].iloc[0])
    fpr = float(TprFpr['fpr'].iloc[0])
    count = len(np.where(test_scores >= thr)[0])      #Faster than using for loop below    
    cc_ouput = count/len(test_scores)   
    
    if (tpr - fpr) == 0:
        pos_prop = cc_ouput
    else:
        pos_prop = (cc_ouput - fpr)/(tpr - fpr)   #adjusted class proportion
    
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    return pos_prop
