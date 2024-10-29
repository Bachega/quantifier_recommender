from _base_recommender import BaseRecommender

from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd
import numpy as np

class KNNRecommender(BaseRecommender):
    def __init__(self, supervised: bool = True, recommender_model = KNeighborsClassifier, _load: bool = True):
        super().__init__(supervised, _load)
        if _load == False:
            self._recommender_model = recommender_model
            self._unscaled_meta_features_table = None
            self.meta_features_table = None
            self.arr_table = None
            
    def fit(self, complete_data_path: str, train_data_path: str, test_data_path: str) -> None:
        dataset_list = [csv for csv in os.listdir(complete_data_path) if csv.endswith(".csv")]
        evaluation_list = []
        for i, dataset in enumerate(dataset_list):
            dataset_name = dataset.split(".csv")[0]
            
            # Meta-Features extraction
            dt = pd.read_csv(complete_data_path + dataset)
            dt = dt.dropna()
            
            if self.supervised:
                y = dt.pop('class')
            else:
                y = None
            X = dt
            
            self._unscaled_meta_features_table = self._extract_and_append(dataset_name, X, y, self._unscaled_meta_features_table)

            # Quantifiers evaluation
            X_train, y_train, X_test, y_test = self._load_train_test_data(dataset_name, train_data_path, test_data_path)
            evaluation_list.append(self.quantifier_evaluator.evaluate_quantifiers(dataset_name,
                                                                                    X_train,
                                                                                    y_train,
                                                                                    X_test,
                                                                                    y_test))
            # DELETE THIS
            if i == 2:
                break

            self.meta_features_table = self._get_normalized_meta_features_table(self._unscaled_meta_features_table)
        

    def recommend(self, X):
        pass