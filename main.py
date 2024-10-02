import pandas as pd
import numpy as np
import os
import pdb
import time
import timeit

from quantifier_recommender import QuantifierRecommender
from quantifier_evaluator import QuantifierEvaluator
from utils_ import generate_train_test_data
from quantifier_evaluator import QuantifierEvaluator

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import is_regressor

from k_quantifier import KQuantifier





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





from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR

def load_train_test_data(dataset_name):
    train_df = pd.read_csv(f"./data/train_data/{dataset_name}.csv")
    y_train = train_df.pop(train_df.columns[-1])
    X_train = train_df

    test_df = pd.read_csv(f"./data/test_data/{dataset_name}.csv")
    y_test = test_df.pop(test_df.columns[-1])
    X_test = test_df

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

def load_test_data(dataset_name, supervised = True):
    test_df = pd.read_csv(f"./data/test_data/{dataset_name}.csv")
    y_test = test_df.pop(test_df.columns[-1])
    X_test = test_df

    if supervised:
        return X_test.to_numpy(), y_test.to_numpy()
    
    return X_test.to_numpy()

def run_quantifiers(dataset_name: str):
    X_train, y_train, X_test, y_test = load_train_test_data(dataset_name=dataset_name)

    clf = LogisticRegression(random_state=42, n_jobs=-1)
    calib_clf = CalibratedClassifierCV(clf, cv=3, n_jobs=-1)
    calib_clf.fit(X_train, y_train)
    scores = getTrainingScores(X_train, y_train, 10, clf)[0]
    pos_scores = scores[scores["class"]==1]["scores"]
    neg_scores = scores[scores["class"]==0]["scores"]
    tprfpr = getTPRFPR(scores)
    clf.fit(X_train, y_train)
    test_scores = clf.predict_proba(X_test)[:,1]

    report = ""
    # ACC
    res = ACC(test_scores, tprfpr)
    report += f"ACC: {res}\n"

    # CC
    res = classify_count(test_scores, thr=0.5)
    report += f"CC: {res}\n"

    # DyS
    res = dys_method(pos_scores, neg_scores, test_scores, measure="hellinger")
    report += f"DyS: {res}\n"

    # HDy
    res = Hdy(pos_scores, neg_scores, test_scores)
    report += f"HDy: {res}\n"

    # MAX
    res = Max(test_scores, tprfpr)
    report += f"MAX: {res}\n"
    
    # MS
    res = MS_method(test_scores, tprfpr)
    report += f"MS: {res}\n"
    
    # PACC
    res = PACC(calib_clf, X_test, tprfpr, thr=0.5)
    report += f"PACC: {res}\n"

    # PCC
    res = PCC(calib_clf, X_test, thr=0.5)
    report += f"PCC: {res}\n"

    # SMM
    res = SMM(pos_scores, neg_scores, test_scores)
    report += f"SMM: {res}\n"

    # SORD
    res = SORD_method(pos_scores, neg_scores, test_scores)
    report += f"SORD: {res}\n"

    # X
    res = X(test_scores, tprfpr)
    report += f"X: {res}\n"

    report += f"TRUE POSITIVE PREVALENCE: {np.count_nonzero(y_test == 1) / len(y_test)}"

    return report

if __name__ == "__main__":
    # recommender = QuantifierRecommender(supervised=True)
    # recommender.load_meta_table_and_fit("recommender_data/meta_features_table.csv", "recommender_data/evaluation_table.csv")
    
    # print(recommender.predict(X_test, y_test))
    dataset = "winetype"

    report = run_quantifiers(dataset)
    k_quantifier = KQuantifier(k=3)
    X_train, y_train, X_test, y_test = load_train_test_data(dataset)
    k_quantifier.fit(X_train, y_train)
    result = k_quantifier.predict(X_test)
    report += f"\nKQUANTIFIER: {result}"
    print(report)

    # # execution_time = timeit.timeit(predict_wrapper, number=20)
    # # print(f"Average execution time over 1000 runs: {execution_time / 20} s")
    
    # start = time.perf_counter()
    # ranking = q.predict(X_test, y_test)
    # stop = time.perf_counter()
    
    # print(f"Time: {stop - start} s")