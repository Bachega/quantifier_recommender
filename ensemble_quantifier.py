import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier

import pdb

class EnsembleQuantifier:
    def __init__(self, ranking: list = None, weights: list = None, method: str ="median") -> None:
        if method not in ["median", "weighted"]:
            raise ValueError("Method must be 'median' or 'weighted'")
        
        # self.k = k
        if ranking is None:
            self.__ranking = None
        else:
            self.ranking = ranking
        self.weights = weights
        self.__clf = None
        self.__calib_clf = None
        self.__scores = None
        self.__pos_scores = None
        self.__neg_scores = None
        self.__tprfpr = None
        self.method = method

    # @property
    # def k(self):
    #     return self.__k
    
    # @k.setter
    # def k(self, k):
    #     if not isinstance(k, int):
    #         raise TypeError("k must be an integer")
        
    #     if k == 0 or k < -1:
    #         raise ValueError("k needs to be a positive number or -1 (to select all quantifiers)")
        
    #     self.__k = k
    
    @property
    def ranking(self):
        return self.__ranking

    @ranking.setter
    def ranking(self, ranking):
        assert isinstance(ranking, list) or isinstance(ranking, tuple), "ranking must be a list/tuple of quantifiers (list of str)."
        self.__ranking = ranking
    
    @property
    def weights(self):
        return self.__weights
    
    @weights.setter
    def weights(self, weights):
        if weights is None or weights == []:
            self.__weights = [] if self.__ranking is None else [1] * len(self.__ranking)
        else:
            assert isinstance(weights, list) or isinstance(weights, tuple), "weights must be a list/tuple of floats."
            self.__weights = weights
    
    @property
    def method(self):
        return self.__method
    
    @method.setter
    def method(self, method):
        if method not in ["median", "weighted"]:
            raise ValueError("Method must be 'median' or 'weighted'")
        self.__method = method
    
    def __str__(self):
        return f"EnsembleQuantifier(ranking={self.ranking}, weights={self.weights}, method={self.method})"
    
    def fit(self, X_train, y_train):
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise TypeError("X_train and y_train must be numpy arrays")
        
        ###### VERIFICAR ######
        self.__scaler = StandardScaler()
        X_train = self.__scaler.fit_transform(X_train)
        ###### VERIFICAR ######

        self.__clf = LogisticRegression(random_state=42, n_jobs=-1)
        self.__calib_clf = CalibratedClassifierCV(self.__clf, cv=3, n_jobs=-1)
        self.__calib_clf.fit(X_train, y_train)
        self.__scores = getTrainingScores(X_train, y_train, 10, self.__clf)[0]
        self.__pos_scores = self.__scores[self.__scores["class"]==1]["scores"]
        self.__neg_scores = self.__scores[self.__scores["class"]==0]["scores"]
        self.__tprfpr = getTPRFPR(self.__scores)
        self.__clf.fit(X_train, y_train)

    def predict(self, X_test):
        assert self.ranking is not None, "The ranking of quantifiers must be provided. Set the 'ranking' attribute."

        ###### VERIFICAR ######
        X_test = self.__scaler.transform(X_test)
        ###### VERIFICAR ######
        
        if self.__method == "median":
            return self.__median_method(X_test)
        elif self.__method == "weighted":
            return self.__weighted_method(X_test)        

    def __median_method(self, X_test):
        test_scores = self.__clf.predict_proba(X_test)[:,1]
        quantifiers = self.__ranking

        predicted_prevalence_list = []
        for quantifier in quantifiers:
            predicted_prevalence_list.append(apply_quantifier(qntMethod=quantifier,
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
    
    def __weighted_method(self, X_test):
        assert len(self.__weights) == len(self.__ranking), "The number of weights in 'weights' must be equal to the number of quantifiers in 'ranking'."
        test_scores = self.__clf.predict_proba(X_test)[:,1]
        quantifiers = self.__ranking
        weight_list = self.__weights

        final_predicted_prevalence = 0
        i = 0
        for quantifier in quantifiers:
            predicted_prevalence = apply_quantifier(qntMethod=quantifier,
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
                                                    external_qnt=None)
            final_predicted_prevalence += weight_list[i] * predicted_prevalence
            i += 1
        return final_predicted_prevalence

    def evaluation(self, recommender_type, recommender_evaluation, quantifiers_evaluation, k_evaluation_path: str = None):
        assert recommender_type == "regression" or recommender_type == "knn", "recommender_type must be 'regression' or 'knn'."

        ensemble_quantifier_eval = pd.DataFrame(columns=["quantifier", "dataset", "sample_size", "sampling_seed",
                                          "iteration", "alpha", "pred_prev", "abs_error", "run_time"])
        for dataset in quantifiers_evaluation["dataset"].unique().tolist():
            ranking = recommender_evaluation.loc[dataset]["predicted_ranking"]
            rows_by_dataset = quantifiers_evaluation[quantifiers_evaluation["dataset"] == dataset]
            alphas = rows_by_dataset["alpha"].unique().tolist()
            iterations = rows_by_dataset["iteration"].unique().tolist()

            for k in range(1, len(ranking) + 1):
                for alph in alphas:
                    for iter in iterations:
                        predicted_prev_list = []
                        run_time_sum = 0
                        sample_size = 0
                        sampling_seed = 0

                        if recommender_type == "regression":
                            error_list = recommender_evaluation.loc[dataset]['predicted_ranking_mae'][:k]
                            if np.any(error_list == 0):
                                error_list = np.array([1e-6 if x == 0 else x for x in error_list])
                            denominator = sum([1/err for err in error_list])
                            weight_list = [(1/err)/denominator for err in error_list]
                            recommender_type_ = "REG"
                        elif recommender_type == "knn":
                            arr_list = recommender_evaluation.loc[dataset]['predicted_ranking_arr'][:k]
                            weight_list = [arr/sum(arr_list) for arr in arr_list]
                            recommender_type_ = "KNN"
                        for i in range(0, k):
                            row = rows_by_dataset[(rows_by_dataset["alpha"] == alph) & (rows_by_dataset["quantifier"] == ranking[i]) & (rows_by_dataset["iteration"] == iter)]
                            sampling_seed = row["sampling_seed"].values[0]
                            predicted_prev_list.append(row["pred_prev"].values[0])
                            run_time_sum += row["run_time"].values[0]
                            sample_size = row["sample_size"].values[0]
                        # MEDIAN METHOD
                        ensemble_quantifier_row = {"quantifier": "("+recommender_type_+")Top-" + str(k),
                                            "dataset": dataset,
                                            "sample_size": sample_size,
                                            "sampling_seed": sampling_seed,
                                            "iteration": iter,
                                            "alpha": alph,
                                            "pred_prev": np.median(predicted_prev_list),
                                            "abs_error": np.abs(np.median(predicted_prev_list) - alph),
                                            "run_time": run_time_sum}
                        ensemble_quantifier_eval.loc[len(ensemble_quantifier_eval)] = ensemble_quantifier_row

                        # WEIGHTED METHOD
                        ensemble_quantifier_row = {"quantifier": "("+recommender_type_+")Top-" + str(k) + "+W",
                                            "dataset": dataset,
                                            "sample_size": sample_size,
                                            "sampling_seed": sampling_seed,
                                            "iteration": iter,
                                            "alpha": alph,
                                            "pred_prev": np.sum(np.array(predicted_prev_list) * np.array(weight_list)),
                                            "abs_error": np.abs(np.sum(np.array(predicted_prev_list) * np.array(weight_list)) - alph),
                                            "run_time": run_time_sum}
                        ensemble_quantifier_eval.loc[len(ensemble_quantifier_eval)] = ensemble_quantifier_row
    
        ensemble_quantifier_eval.sort_values(by=['quantifier', 'dataset'], inplace=True)
        ensemble_quantifier_eval.reset_index(drop=True, inplace=True)
        if k_evaluation_path is not None:
            ensemble_quantifier_eval.to_csv(k_evaluation_path, index=False)
            
        return ensemble_quantifier_eval