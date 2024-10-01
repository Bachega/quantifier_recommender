from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


from quantifier_recommender import QuantifierRecommender
from utils import getTrainingScores, getTPRFPR
from utils.applyquantifiers import apply_quantifier

import pdb

class KQuantifier:
    def __init__(self, k = 1) -> None:
        self.k = k
        self.__quantifier_recommender = QuantifierRecommender(supervised=True)
        
        self.__clf = None
        self.__calib_clf = None
        self.__scores = None
        self.__tprfpr = None
        self.__pos_scores = None
        self.__neg_scores = None

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
        meta_features_path = "recommender_data/meta_features_table.csv"
        evaluation_table_path = "recommender_data/evaluation_table.csv"
        self.__quantifier_recommender.load_and_fit_meta_table(meta_features_path, evaluation_table_path)
        
        clf = LogisticRegression(random_state=42, n_jobs=-1)
        calib_clf = CalibratedClassifierCV(clf, cv=3, n_jobs=-1)
        calib_clf.fit(X_train, y_train)
        scores = getTrainingScores(X_train, y_train, 10, clf)[0]
        tprfpr = getTPRFPR(scores)
        clf.fit(X_train, y_train)

        # for index, value in self.__quantifier_recommender.predict(X_train, y_train, k=self.k):
        
        # for 

        predicted_positive_prevalence = apply_quantifier(qntMethod=quantifier,
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
                                                         test_quapy=None,
                                                         external_qnt=None) #y_test=test_label

    def predict(self, X):
        pass

        # ranking = self.__quantifier_recommender.predict(X, k=self.k)
        