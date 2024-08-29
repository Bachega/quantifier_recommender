from quantifiers.CC import classify_count
from quantifiers.ACC import ACC
from quantifiers.PCC import PCC
from quantifiers.PACC import PACC
from quantifiers.HDy import Hdy
from quantifiers.X import X
from quantifiers.MAX import Max
from quantifiers.SMM import SMM  
from quantifiers.dys_method import dys_method
from quantifiers.sord import SORD_method
from quantifiers.MS import MS_method
from quantifiers.MS_2 import MS_method2
from quantifiers.T50 import T50
from quantifiers.PWK import PWK
from quantifiers.GAC import GAC
from quantifiers.GPAC import GPAC
from quantifiers.FM import FM
#from quantifiers.emq_quapy import EMQ_quapy
from utils.predictQuantifierSchumacherGithub import predictQuantifierSchumacherGithub
import numpy as np
import pandas as pd
import pdb
import time

def apply_quantifier(qntMethod, 
                    clf,
                    scores, 
                    p_score, 
                    n_score, 
                    train_labels, 
                    test_score, 
                    TprFpr, 
                    thr, 
                    measure,                     
                    test_data,
                    test_quapy, 
                    external_qnt):
    """This function is an interface for running different quantification methods.
 
    Parameters
    ----------
    qntMethod : string
        Quantification method name
    p_score : array
        A numeric vector of positive scores estimated either from a validation set or from a cross-validation method.
    n_score : array
        A numeric vector of negative scores estimated either from a validation set or from a cross-validation method.
    test : array
        A numeric vector of scores predicted from the test set.
    TprFpr : matrix
        A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
    thr : float
        The threshold value for classifying and counting. Default is 0.5.
    measure : string
        Dissimilarity function name used by the DyS method. Default is "topsoe".
    calib_clf : object
        A calibrated classifier used when PCC or PACC methods are called by the main experimental setup.
    te_data : dataframe
        A dataframe of test data.
    pwk_clf : object
        Nearest Neighbour classifier used only when PWK method is called. 
    schumacher_qnt : class
        contains functions of quantification methods used in schumacher paper
    test_quapy : dataframe
        A dataframe of test sample for Quapy package.
    model_quapy : object
        EMQ Model fitted using Quapy package (base RF calssifier)
    Returns
    -------
    array
        the class distribution of the test calculated according to the qntMethod quantifier. 
    """
    
    external_quantifiers = ['Readme', 'HDx', 'FormanMM', 'CDE', 'PWK'] #quantifiers from schumacher paper
    # print(qntMethod)
    if qntMethod in external_quantifiers:  
        return predictQuantifierSchumacherGithub(external_qnt, test_data)
    if qntMethod == "CC":
        return classify_count(test_score, thr)
    if qntMethod == "ACC":        
        return ACC(test_score, TprFpr)
    if qntMethod == "EM":         # This EMQ method is used from Quapy packages
        pos_prop = external_qnt.quantify(test_quapy.instances)  
        return pos_prop[1]    
    if qntMethod == "SMM":
        return SMM(p_score, n_score, test_score)
    if qntMethod == "HDy":
        return Hdy(p_score, n_score, test_score)
    if qntMethod == "DyS":
        return dys_method(p_score, n_score, test_score,measure)
    if qntMethod == "SORD":
        return SORD_method(p_score, n_score,test_score)
    if qntMethod == "MS":
        return MS_method(test_score, TprFpr)
    if qntMethod == "MS2":
        return MS_method2(test_score, TprFpr)
    if qntMethod == "MAX":
        return Max(test_score, TprFpr)
    if qntMethod == "X":
        return X(test_score, TprFpr)
    if qntMethod == "T50":
        return T50(test_score, TprFpr)
    if qntMethod == "PCC":
        return PCC(clf, test_data,thr)
    if qntMethod == "PACC":
        return PACC(clf, test_data, TprFpr, thr)
    # if qntMethod == "PWK":
    #     return PWK(test_data, external_qnt)
    if qntMethod == "GAC":
        sc_p = np.append(np.array(p_score), n_score)
        sc_n = 1-sc_p
        scores = np.array(pd.concat([pd.DataFrame(sc_p), pd.DataFrame(sc_n)], axis=1))    
        sc_te = np.array(pd.concat([pd.DataFrame(test_score), pd.DataFrame(1-test_score)], axis=1))
        return GAC(scores, sc_te, np.array(train_labels), 2)
    if qntMethod == "GPAC":
        sc_p = np.append(np.array(p_score), n_score)
        sc_n = 1-sc_p
        scores = np.array(pd.concat([pd.DataFrame(sc_p), pd.DataFrame(sc_n)], axis=1))
        sc_te = np.array(pd.concat([pd.DataFrame(test_score), pd.DataFrame(1-test_score)], axis=1))        
        prop = GPAC(scores, sc_te, np.array(train_labels), 2)        
        return prop
    if qntMethod == "FM":  
        sc_p = np.append(np.array(p_score), n_score)        
        sc_n = 1-sc_p
        scores = np.array(pd.concat([pd.DataFrame(sc_p), pd.DataFrame(sc_n)], axis=1))
        sc_te = np.array(pd.concat([pd.DataFrame(test_score), pd.DataFrame(1-test_score)], axis=1))
        prop = FM(scores, sc_te, np.array(train_labels), 2)
        return prop
