import pandas as pd
import numpy as np
import os
import pdb
import time
import timeit

from quantifier_recommender import QuantifierRecommender
from quantifier_evaluator import QuantifierEvaluator
from k_quantifier import KQuantifier

from utils_ import generate_train_test_data


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import is_regressor



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
from evaluate_quantifiers_ import evaluate_quantifiers

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
    start = time.perf_counter()
    res = ACC(test_scores, tprfpr)
    stop = time.perf_counter()
    report += f"ACC: {res} | Runtime: {stop - start} s\n"

    # CC
    start = time.perf_counter()
    res = classify_count(test_scores, thr=0.5)
    stop = time.perf_counter()
    report += f"CC: {res} | Runtime: {stop - start} s\n"

    # DyS
    start = time.perf_counter()
    res = dys_method(pos_scores, neg_scores, test_scores, measure="hellinger")
    stop = time.perf_counter()
    report += f"DyS: {res} | Runtime: {stop - start} s\n"

    # HDy
    start = time.perf_counter()
    res = Hdy(pos_scores, neg_scores, test_scores)
    stop = time.perf_counter()
    report += f"HDy: {res} | Runtime: {stop - start} s\n"

    # MAX
    start = time.perf_counter()
    res = Max(test_scores, tprfpr)
    stop = time.perf_counter()
    report += f"MAX: {res} | Runtime: {stop - start} s\n"
    
    # MS
    start = time.perf_counter()
    res = MS_method(test_scores, tprfpr)
    stop = time.perf_counter()
    report += f"MS: {res} | Runtime: {stop - start} s\n"
    
    # PACC
    start = time.perf_counter()
    res = PACC(calib_clf, X_test, tprfpr, thr=0.5)
    stop = time.perf_counter()
    report += f"PACC: {res} | Runtime: {stop - start} s\n"

    # PCC
    start = time.perf_counter()
    res = PCC(calib_clf, X_test, thr=0.5)
    stop = time.perf_counter()
    report += f"PCC: {res} | Runtime: {stop - start} s\n"

    # SMM
    start = time.perf_counter()
    res = SMM(pos_scores, neg_scores, test_scores)
    stop = time.perf_counter()
    report += f"SMM: {res} | Runtime: {stop - start} s\n"

    # SORD
    start = time.perf_counter()
    res = SORD_method(pos_scores, neg_scores, test_scores)
    stop = time.perf_counter()
    report += f"SORD: {res} | Runtime: {stop - start} s\n"

    # X
    start = time.perf_counter()
    res = X(test_scores, tprfpr)
    stop = time.perf_counter()
    report += f"X: {res} | Runtime: {stop - start} s\n"

    report += f"TRUE POSITIVE PREVALENCE: {np.count_nonzero(y_test == 1) / len(y_test)}"

    return report

if __name__ == "__main__":
    
    # recommender = QuantifierRecommender(supervised=True)
    # recommender.load_fit_meta_table("./recommender_data/meta_table.h5")
    X_test, y_test = load_test_data("anuranCalls", supervised=True)
    print(recommender.predict(X_test, y_test))
    
    
    # recommender.save_meta_table()
    # print("Starting...")
    # start = time.perf_counter()
    # recommender = QuantifierRecommender(supervised=True)
    # recommender.load_fit_meta_table("./recommender_data/meta_table.h5")
    # stop = time.perf_counter()

    # print(f"Time: {stop - start} s")
    
    # recommender.fit(datasets_path="./datasets/", train_data_path="./data/train_data/", test_data_path="./data/test_data/")
    # recommender.save_meta_table("recommender_data/meta_features_table.csv", "recommender_data/evaluation_table.csv")
    
    # recommender = QuantifierRecommender(supervised=True)
    # recommender.load_meta_table_fit("recommender_data/meta_features_table.csv", "recommender_data/evaluation_table.csv")
    
    
    # print(recommender.predict(X_test, y_test))





    # dataset = "anuranCalls"

    # X_train, y_train, X_test, y_test = load_train_test_data(dataset)
    # eval_table = evaluate_quantifiers(dataset, X_train, y_train, X_test, y_test)
    # eval_table.to_csv("./evaluation_table.csv", index=False)








    # report = ""
    # report += run_quantifiers(dataset)
    # k_quantifier = KQuantifier(k=-1)
    # X_train, y_train, X_test, y_test = load_train_test_data(dataset)
    # k_quantifier.fit(X_train, y_train)
    
    # start = time.perf_counter()
    # result = k_quantifier.predict(X_test)
    # stop = time.perf_counter()

    # report += f"\nKQUANTIFIER: {result} | Runtime: {stop - start} s\n"
    # print(report)

    # # execution_time = timeit.timeit(predict_wrapper, number=20)
    # # print(f"Average execution time over 1000 runs: {execution_time / 20} s")
    
    # start = time.perf_counter()
    # ranking = q.predict(X_test, y_test)
    # stop = time.perf_counter()
    
    # print(f"Time: {stop - start} s")