import os
import pandas as pd
import numpy as np
import time
from copy import deepcopy

from mlquantify.methods import CC, ACC, MAX, PCC, PACC, X_method, MS, MS2, HDy, SMM, SORD, DyS, PWK, T50, EMQ
from mlquantify.classification import PWKCLF
from mlquantify import set_arguments
from mlquantify.methods import AGGREGATIVE

from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

class QuantifierEvaluator:
    __remove = ['GAC', 'GPAC', 'FM', 'DySsyn']
    
    __qtf_evaluation_table_columns = ["quantifier",
                                      "dataset",
                                      "sample_size",
                                      "sampling_seed",
                                      "iteration",
                                      "alpha",
                                      "pred_prev",
                                      "none",
                                      "run_time"]
    
    def __init__(self) -> None:
        self.__quantifiers = deepcopy(AGGREGATIVE)
        for __qtf in self.__remove:
            self.__quantifiers.pop(__qtf, None)
        
        self.qtf_evaluation_table = None
        
        # int(91243342813999289490899206158518312834)
        self.__seed = int(91243342813999289490899206158518312834)
        self.__rng = np.random.default_rng(seed=self.__seed)

    def __append_to_qtf_evaluation_table(self, quantifier, dataset_name, sample_size, sampling_seed, iteration, alpha, pred_prev, perf_metric, run_time):        
        self.qtf_evaluation_table.loc[len(self.qtf_evaluation_table.index)] = [quantifier,
                                                                               dataset_name,
                                                                               sample_size,
                                                                               sampling_seed,
                                                                               iteration,
                                                                               alpha,
                                                                               pred_prev,
                                                                               perf_metric,
                                                                               run_time]
    
    def __generate_seed(self):
        return int(self.__rng.random() * 1000000)
    
    def __reset_seed_generator(self):
        self.__rng = np.random.default_rng(seed=self.__seed)

    def evaluate_quantifiers(self, dataset_name, X_train, y_train, X_test, y_test, quantifiers = None, func_type: str = "cost"):
        assert func_type in ["cost", "utility"], "Argument 'func_type' needs to be either 'cost' or 'utility'"

        if func_type == "cost":
            self.__qtf_evaluation_table_columns[7] = "abs_error"
        elif func_type == "utility":
            self.__qtf_evaluation_table_columns[7] = "inv_abs_error"
        self.qtf_evaluation_table = pd.DataFrame(columns=self.__qtf_evaluation_table_columns)
        
        if quantifiers is None:
            quantifiers = self.__quantifiers
        else:
            if not isinstance(quantifiers, list):
                raise TypeError("Argument 'quantifiers' needs to be a 'list' of quantifiers to evaluate")

            if not set(quantifiers).issubset(list(self.__quantifiers.keys())):
                raise ValueError(f"List of quantifiers contains invalid values (like names of non implemented quantifiers). Available quantifiers are {list(self.__quantifiers.keys())}")
            
        train_scores, clf = getTrainingScores(X_train, y_train, 10, LogisticRegression(random_state=42))
        y_labels = train_scores.pop('class').to_numpy()
        
        niterations = 10
        batch_sizes = list([100])
        alpha_values = [round(x, 2) for x in np.linspace(0,1,20)]

        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test, columns=[str(len(X_test.columns))])
        df_test = pd.concat([X_test, y_test], axis=1)

        df_test_pos = df_test.loc[df_test[df_test.columns[-1]] == 1]
        df_test_neg = df_test.loc[df_test[df_test.columns[-1]] == 0]

        for sample_size in batch_sizes:
            for alpha in alpha_values:
                # Repeats the same experiment (to reduce variance)
                for iter in range(niterations):
                    pos_size = int(round(sample_size * alpha, 2))
                    neg_size = sample_size - pos_size

                    sampling_seed = self.__generate_seed()

                    sample_test_pos = df_test_pos.sample( int(pos_size), replace = False, random_state=sampling_seed )
                    sample_test_neg = df_test_neg.sample( int(neg_size), replace = False, random_state=sampling_seed )

                    sample_test = pd.concat([sample_test_pos, sample_test_neg])
                    sample_test = sample_test.sample(frac=1, random_state=sampling_seed).reset_index(drop=True)

                    y_test = sample_test.iloc[:, -1].to_numpy()
                    X_test = sample_test.iloc[:, :-1].to_numpy()

                    test_label = sample_test[sample_test.columns[-1]]
                    n_pos_sample_test = list(test_label).count(1) #Counting num of actual positives in test sample
                    calcultd_pos_prop = round(n_pos_sample_test/len(sample_test), 2) #actual pos class prevalence in generated sample

                    test_scores = clf.predict_proba(X_test)# [:,1]
                    y_pred = clf.predict(X_test)

                    set_arguments(y_pred=y_pred.tolist(), # Predictions for the TEST set ()
                            posteriors_test=test_scores, # Scores of the TEST set ()
                            y_labels=y_labels.tolist(), # True labels of the TRAIN set ()
                            posteriors_train=train_scores # Scores extracted from TRAIN set ()
                        )
                    
                    for key, value in self.__quantifiers.items():
                        if key == "PWK":
                            qtf = value(learner=PWKCLF())
                        else:
                            qtf = value()

                        start = time.perf_counter()
                        
                        qtf.fit(X_train, y_train)
                        pred_dict = qtf.predict(X_test)
                        pred_pos_prop = pred_dict[1] if 1 in pred_dict else 0.0
                        
                        stop = time.perf_counter()
                        run_time = stop - start

                        if func_type == "cost":
                            perf_metric = round(abs(calcultd_pos_prop - pred_pos_prop), 2)
                        elif func_type == "utility":
                            perf_metric = round((1 / (1 + abs(calcultd_pos_prop - pred_pos_prop))), 2)

                        self.__append_to_qtf_evaluation_table(key,
                                                              dataset_name,
                                                              sample_size,
                                                              sampling_seed,
                                                              iter+1,
                                                              alpha,
                                                              pred_pos_prop,
                                                              perf_metric,
                                                              run_time)
        self.__reset_seed_generator()
        return self.qtf_evaluation_table