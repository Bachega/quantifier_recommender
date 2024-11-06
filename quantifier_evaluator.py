import pandas as pd
import numpy as np
import time
import joblib

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
                                      "sampling_seed",
                                      "iteration",
                                      "alpha",
                                      "pred_prev",
                                      "none",
                                      "run_time"]
    
    def __init__(self, _regenerate_seeds = False) -> None:
        self.qtf_evaluation_table = None 

        if _regenerate_seeds:
            self.__seeds = self.__generate_seeds()
            self.__save_seeds()
        else:
            self.__seeds = self.__load_seeds()

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
    
    def __generate_seeds(self, size: int = 1000):
        seeds = np.random.choice(np.arange(size), size=int(size), replace=False)
        seeds = seeds.astype(int)
        return seeds
    
    def __save_seeds(self, path: str = "./evaluator_data/seeds.npy"):
        np.save(path, self.__seeds)
    
    def __load_seeds(self, path: str = "./evaluator_data/seeds.npy"):
        return np.load(path)

    def __get_seed(self, index):
        return self.__seeds[index]

    def evaluate_quantifiers(self, dataset_name, X_train, y_train, X_test, y_test, quantifiers = None, func_type: str = "cost"):
        assert func_type in ["cost", "utility"], "Argument 'func_type' needs to be either 'cost' or 'utility'"

        if func_type == "cost":
            self.__qtf_evaluation_table_columns[7] = "abs_error"
        elif func_type == "utility":
            self.__qtf_evaluation_table_columns[7] = "inv_abs_error"
        self.qtf_evaluation_table = pd.DataFrame(columns=self.__qtf_evaluation_table_columns)

        if quantifiers is None:
            quantifiers = self.__quantifiers
        
        if not isinstance(quantifiers, list):
            raise TypeError("Argument 'quantifiers' needs to be a 'list' of quantifiers to evaluate")

        if not set(quantifiers).issubset(self.__quantifiers):
            raise ValueError(f"List of quantifiers contains invalid values (like names of non implemented quantifiers). Available quantifiers are {self.__quantifiers}")

        clf = None
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

        seed_index = 0
        for sample_size in batch_sizes:
            for alpha in alpha_values:
                # Repeats the same experiment (to reduce variance)
                for iter in range(niterations):
                    pos_size = int(round(sample_size * alpha, 2))
                    neg_size = sample_size - pos_size
                    
                    sampling_seed = self.__get_seed(seed_index)
                    seed_index += 1

                    sample_test_pos = df_test_pos.sample( int(pos_size), replace = False, random_state=sampling_seed)
                    sample_test_neg = df_test_neg.sample( int(neg_size), replace = False, random_state=sampling_seed)
                    
                    sample_test = pd.concat([sample_test_pos, sample_test_neg])
                    test_label = sample_test[sample_test.columns[-1]]

                    test_sample = sample_test.drop([sample_test.columns[-1]], axis=1) # sample_test.drop(["class"], axis=1)  #dropping class label columns
                    te_scores = clf.predict_proba(test_sample)[:,1]  #estimating test sample scores

                    n_pos_sample_test = list(test_label).count(1) #Counting num of actual positives in test sample
                    calcultd_pos_prop = round(n_pos_sample_test/len(sample_test), 2) #actual pos class prevalence in generated sample

                    for quantifier in quantifiers:
                        #.............Calling of Methods..................................................
                        start = time.perf_counter()
                        pred_pos_prop = apply_quantifier(qntMethod=quantifier,
                                                        clf=calib_clf,
                                                        scores=scores,
                                                        p_score=pos_scores,
                                                        n_score=neg_scores,
                                                        train_labels=None,
                                                        test_score=te_scores,
                                                        TprFpr=tprfpr,
                                                        thr=0.5,
                                                        measure='topsoe',
                                                        test_data=test_sample,
                                                        test_quapy=None,
                                                        external_qnt=None)
                        stop = time.perf_counter()
                        run_time = stop - start
                        # pred_pos_prop = np.round(pred_pos_prop, 2)  #predicted class proportion
                        #..............................RESULTS Evaluation.....................................
                        if func_type == "cost":
                            perf_metric = round(abs(calcultd_pos_prop - pred_pos_prop), 2)
                        elif func_type == "utility":
                            perf_metric = round((1 / (1 + abs(calcultd_pos_prop - pred_pos_prop))), 2)
                        # abs_error = round(abs(calcultd_pos_prop - pred_pos_prop), 2) # absolute error

                        self.__append_to_qtf_evaluation_table(quantifier,
                                                              dataset_name,
                                                              sample_size,
                                                              sampling_seed,
                                                              iter+1,
                                                              alpha,
                                                              pred_pos_prop,
                                                              perf_metric,
                                                              run_time)
        # self.__sort_qtf_evaluation_table()
        return self.qtf_evaluation_table