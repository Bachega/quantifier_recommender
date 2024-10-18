import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from quantifier_recommender import QuantifierRecommender
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier

import pdb

class KQuantifier:
    def __init__(self, k: int = 1, method: str ="median") -> None:
        self.k = k
        self.__quantifier_recommender = QuantifierRecommender(supervised=True)
        self.__clf = None
        self.__calib_clf = None
        self.__scores = None
        self.__pos_scores = None
        self.__neg_scores = None
        self.__tprfpr = None
        self.__method = method

    @property
    def k(self):
        return self.__k
    
    @k.setter
    def k(self, k):
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        if k == 0 or k < -1:
            raise ValueError("k needs to be a positive number or -1 (to select all quantifiers)")
        
        self.__k = k
    
    def fit(self, X_train, y_train):
        # self.__quantifier_recommender.load_fit_meta_table("./recommender_data/meta_table.h5")
        self.__k_quantifiers = self.__quantifier_recommender.predict(X_train, y_train, k=self.k)
        self.__clf = LogisticRegression(random_state=42, n_jobs=-1)
        self.__calib_clf = CalibratedClassifierCV(self.__clf, cv=3, n_jobs=-1)
        self.__calib_clf.fit(X_train, y_train)
        self.__scores = getTrainingScores(X_train, y_train, 10, self.__clf)[0]
        self.__pos_scores = self.__scores[self.__scores["class"]==1]["scores"]
        self.__neg_scores = self.__scores[self.__scores["class"]==0]["scores"]
        self.__tprfpr = getTPRFPR(self.__scores)
        self.__clf.fit(X_train, y_train)

    def predict(self, X_test):
        if self.__method == "median":
            return self.__median_method(X_test)
        elif self.__method == "weighted":
            return self.weighted_method(X_test)
        else:
            raise ValueError("Method must be 'mean' or 'median'")

    def __median_method(self, X_test):
        test_scores = self.__clf.predict_proba(X_test)[:,1]

        predicted_prevalence_list = []
        for _, quantifier in self.__k_quantifiers.items():
            predicted_prevalence_list.append(apply_quantifier(qntMethod=quantifier[0],
                                                              clf=self.__calib_clf,
                                                              scores=self.__scores,
                                                              p_score=self.__pos_scores,
                                                              n_score=self.__neg_scores,
                                                              train_labels=None,
                                                              test_score=test_scores,
                                                              TprFpr=self.__tprfpr,
                                                              thr=0.5,
                                                              measure='hellinger',
                                                              test_data=X_test,
                                                              test_quapy=None,
                                                              external_qnt=None))
        return np.median(predicted_prevalence_list)
    
    def evaluation(self, recommender_evaluation, quantifiers_evaluation, k_evaluation_path: str = None):
        k_quantifier_eval = pd.DataFrame(columns=["quantifier", "dataset", "sample_size",
                                          "real_prev", "pred_prev", "abs_error",
                                          "run_time"])
        for dataset in quantifiers_evaluation["dataset"].unique().tolist():
            ranking = recommender_evaluation.loc[dataset]["predicted_ranking"]
            rows_by_dataset = quantifiers_evaluation[quantifiers_evaluation["dataset"] == dataset]
            alphas = rows_by_dataset["real_prev"].unique().tolist()

            for k in range(1, len(ranking) + 1):
                for alph in alphas:
                    predicted_prev_list = []
                    run_time_sum = 0
                    sample_size = 0
                    for i in range(0, k):
                        row = rows_by_dataset[(rows_by_dataset["real_prev"] == alph) & (rows_by_dataset["quantifier"] == ranking[i])]
                        predicted_prev_list.append(row["pred_prev"].values[0])
                        run_time_sum += row["run_time"].values[0]
                        sample_size = row["sample_size"].values[0]
                    pdb.set_trace()
                    k_quantifier_row = {"quantifier": "Top-" + str(k),
                                        "dataset": dataset,
                                        "sample_size": sample_size,
                                        "real_prev": alph,
                                        "pred_prev": np.median(predicted_prev_list),
                                        "abs_error": np.abs(np.median(predicted_prev_list) - alph),
                                        "run_time": run_time_sum}
                    k_quantifier_eval.loc[len(k_quantifier_eval)] = k_quantifier_row
    
        k_quantifier_eval.sort_values(by=['quantifier', 'dataset'], inplace=True)
        k_quantifier_eval.reset_index(drop=True, inplace=True)
        if k_evaluation_path is not None:
            k_quantifier_eval.to_csv(k_evaluation_path, index=False)
            
        return k_quantifier_eval