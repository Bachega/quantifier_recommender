import os
import pandas as pd
import numpy as np
import json
import time
import joblib
import pdb
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

class QuantifierEvaluator:
    __quantifiers = ["ACC",
                     "CC",
                     "DyS",
                     "HDy",
                     "MAX",
                     "MS",
                     "PACC",
                     "PCC",
                     "SMM",
                     "SORD",
                     "X"]
    
    __qtf_evaluation_table_columns = ["quantifier",
                                      "dataset",
                                      "sample_size",
                                      "real_prev",
                                      "pred_prev",
                                      "abs_error",
                                      "run_time"]
    
    def __init__(self) -> None:
        self.qtf_evaluation_table = pd.DataFrame(columns=self.__qtf_evaluation_table_columns)

    def __append_to_qtf_evaluation_table(self, quantifier, dataset_name, sample_size, alpha, pred_pos_prop, abs_error, run_time):        
        self.qtf_evaluation_table.loc[len(self.qtf_evaluation_table.index)] = [quantifier,
                                                                               dataset_name,
                                                                               sample_size,
                                                                               alpha,
                                                                               pred_pos_prop,
                                                                               abs_error,
                                                                               run_time]
    
    def __aggregate_qtf_evaluation_table(self):
        self.qtf_evaluation_table = self.qtf_evaluation_table.groupby(['quantifier', 'dataset'])[["abs_error", "run_time"]].aggregate('mean')

    # def save_evaluation_table(self, path = "./evaluation_table.csv"):
    #     self.sort_evaluation_table()
    #     self.evaluation_table.to_csv(path, index=False)
    
    # def load_evaluation_table(self, path = "./evaluation_table.csv"):
    #     evaluation_table = pd.read_csv(path)
    #     evaluation_table_columns = self.__evaluation_table_columns

    #     i = 0
    #     for column in evaluation_table.columns.to_list():
    #         if column != evaluation_table_columns[i]:
    #             raise Exception(f"Invalid columns.\nLoaded evaluation table needs the following columns: {evaluation_table_columns}.")
    #         i += 1
        
    #     self.evaluation_table = evaluation_table
    
    # def __sort_qtf_evaluation_table(self):
    #     self.qtf_evaluation_table.sort_values(by=['quantifier', 'dataset'], inplace=True)

    def evaluate_internal_quantifiers(self, dataset_name, X_train, y_train, X_test, y_test):
        self.qtf_evaluation_table = self.qtf_evaluation_table.iloc[0:0]
        
        clf = None
        try:
            clf = joblib.load(f"./data/estimator_parameters/{dataset_name}.joblib")
            clf.n_jobs = -1
        except:
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
                abs_error_dict = {key: [] for key in self.__quantifiers}
                run_time_dict = {key: [] for key in self.__quantifiers}
                # pdb.set_trace()

                # Repeats the same experiment (to reduce variance)
                for iter in range(niterations):
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

                    for quantifier in self.__quantifiers:
                        #..............Test Sample QUAPY exp...........................
                        te_quapy = None
                        external_qnt = None

                        
                        #.............Calling of Methods..................................................
                        start = time.time()
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
                        stop = time.time()
                        run_time_dict[quantifier].append(stop - start)

                        pred_pos_prop = np.round(pred_pos_prop,2)  #predicted class proportion
                        
                        #..............................RESULTS Evaluation.....................................
                        abs_error = round(abs(calcultd_pos_prop - pred_pos_prop), 2) # absolute error
                        abs_error_dict[quantifier].append(abs_error)
                
                for quantifier in abs_error_dict.keys():
                    self.__append_to_qtf_evaluation_table(quantifier,
                                                          dataset_name,
                                                          sample_size,
                                                          alpha,
                                                          pred_pos_prop,
                                                          np.mean(abs_error_dict[quantifier]),
                                                          np.min(run_time_dict[quantifier]))
                    abs_error_dict[quantifier].clear()
                    run_time_dict[quantifier].clear()
        
        self.__sort_qtf_evaluation_table()
        return self.qtf_evaluation_table