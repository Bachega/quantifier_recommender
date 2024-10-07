import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from quantifier_recommender import QuantifierRecommender
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier

import pdb

class KQuantifier:
    def __init__(self, k = 1) -> None:
        self.k = k
        self.__quantifier_recommender = QuantifierRecommender(supervised=True)
        self.__clf = None
        self.__calib_clf = None
        self.__scores = None
        self.__pos_scores = None
        self.__neg_scores = None
        self.__tprfpr = None

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
        self.__quantifier_recommender.load_fit_meta_table("./recommender_data/meta_table.h5")

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