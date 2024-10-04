import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier
from sklearn.calibration import CalibratedClassifierCV
from k_quantifier import KQuantifier
import time

def evaluate_quantifiers(dataset_name, X_train, y_train, X_test, y_test, quantifiers = None):
    k_quantifier = KQuantifier(k=3)
    k_quantifier.fit(X_train, y_train)

    quantifiers_ = ["ACC", "CC", "DyS", "HDy", "MAX", "MS", "PACC", "PCC", "SMM", "SORD", "X", "KQuantifier"]

    if quantifiers is None:
        quantifiers = quantifiers_
    
    if not isinstance(quantifiers, list):
        raise TypeError("Argument 'quantifiers' needs to be a 'list' of quantifiers to evaluate")

    if not set(quantifiers).issubset(quantifiers_):
        raise ValueError(f"List of quantifiers contains invalid values (like names of non implemented quantifiers). Available quantifiers are {quantifiers_}")

    evaluation_table = pd.DataFrame(columns=["quantifier", "dataset", "sample_size","real_prev","pred_prev","abs_error","run_time"])
    
    clf = LogisticRegression(random_state=42, n_jobs=-1)
    
    calib_clf = CalibratedClassifierCV(clf, cv=3, n_jobs=-1)
    calib_clf.fit(X_train, y_train)

    scores = getTrainingScores(X_train, y_train, 10, clf)[0]
    tprfpr = getTPRFPR(scores)
    clf.fit(X_train, y_train)

    niterations = 10
    batch_sizes = list([100])
    alpha_values = [round(x, 2) for x in np.linspace(0,1,20)]

    pos_scores = scores[scores["class"]==1]["scores"]
    neg_scores = scores[scores["class"]==0]["scores"]

    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test, columns=[str(len(X_test.columns))])
    df_test = pd.concat([X_test, y_test], axis=1)

    df_test_pos = df_test.loc[df_test[df_test.columns[-1]] == 1]
    df_test_neg = df_test.loc[df_test[df_test.columns[-1]] == 0]

    for sample_size in batch_sizes:
        for alpha in alpha_values:
            abs_error_dict = {key: [] for key in quantifiers_}
            run_time_dict = {key: [] for key in quantifiers_}
            # pdb.set_trace()

            # Repeats the same experiment (to reduce variance)
            for iter in range(1): # ----------------------------> Change this to niterations
            # for iter in range(niterations):
                pos_size = int(round(sample_size * alpha, 2))
                neg_size = sample_size - pos_size
                
                sample_test_pos = df_test_pos.sample( int(pos_size), replace = False)
                sample_test_neg = df_test_neg.sample( int(neg_size), replace = False)
                
                sample_test = pd.concat([sample_test_pos, sample_test_neg])
                test_label = sample_test[sample_test.columns[-1]]

                test_sample = sample_test.drop([sample_test.columns[-1]], axis=1) # sample_test.drop(["class"], axis=1)  #dropping class label columns
                te_scores = clf.predict_proba(test_sample)[:,1]  #estimating test sample scores

                n_pos_sample_test = list(test_label).count(1) #Counting num of actual positives in test sample
                calcultd_pos_prop = round(n_pos_sample_test/len(sample_test), 2) #actual pos class prevalence in generated sample

                for quantifier in quantifiers:
                    #..............Test Sample QUAPY exp...........................
                    te_quapy = None
                    external_qnt = None

                    
                    #.............Calling of Methods..................................................
                    start = time.perf_counter()
                    if not quantifier == "KQuantifier":
                        pred_pos_prop = apply_quantifier(qntMethod=quantifier,
                                                        clf=calib_clf,
                                                        scores=scores,
                                                        p_score=pos_scores,
                                                        n_score=neg_scores,
                                                        train_labels=None,
                                                        test_score=te_scores,
                                                        TprFpr=tprfpr,
                                                        thr=0.5,
                                                        measure='hellinger',
                                                        test_data=test_sample,
                                                        test_quapy=te_quapy,
                                                        external_qnt=external_qnt) #y_test=test_label
                    else:
                        pred_pos_prop = k_quantifier.predict(test_sample)
                    stop = time.perf_counter()
                    run_time_dict[quantifier].append(stop - start)

                    pred_pos_prop = np.round(pred_pos_prop,2)  #predicted class proportion
                    
                    #..............................RESULTS Evaluation.....................................
                    abs_error = round(abs(calcultd_pos_prop - pred_pos_prop), 2) # absolute error
                    abs_error_dict[quantifier].append(abs_error)
            
            for quantifier in abs_error_dict.keys():
                evaluation_table.loc[len(evaluation_table.index)] = [quantifier,
                                                                     dataset_name,
                                                                     sample_size,
                                                                     alpha,
                                                                     pred_pos_prop,
                                                                     np.mean(abs_error_dict[quantifier]),
                                                                     np.min(run_time_dict[quantifier])]
                abs_error_dict[quantifier].clear()
                run_time_dict[quantifier].clear()
    return evaluation_table