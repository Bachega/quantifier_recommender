from base_recommender import BaseRecommender

from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd
import numpy as np

import pdb

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
        for _, dataset in enumerate(dataset_list):
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
            evaluation_list.append(self.quantifier_evaluator.evaluate_quantifiers(dataset_name=dataset_name,
                                                                                  X_train=X_train,
                                                                                  y_train=y_train,
                                                                                  X_test=X_test,
                                                                                  y_test=y_test,
                                                                                  func_type="utility"))
            # DELETE THIS
            # if i == 2:
            break

        # Normalize the extracted meta-features
        self.meta_features_table = self._get_normalized_meta_features_table(self._unscaled_meta_features_table, method="minmax")

        # Concatenate all the evaluations into a single evaluation table
        # and then sort and aggregate the quantifiers evaluations
        self._not_agg_evaluation_table = pd.concat(evaluation_list, axis=0)

        self.evaluation_table = self._not_agg_evaluation_table.sort_values(by=['quantifier', 'dataset'])
        self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset", "alpha"]).agg(
            pred_prev = pd.NamedAgg(column="pred_prev", aggfunc="mean"),
            inv_abs_error = pd.NamedAgg(column="inv_abs_error", aggfunc="mean"),
            sample_size = pd.NamedAgg(column="sample_size", aggfunc="first"),
            sampling_seed = pd.NamedAgg(column="sampling_seed", aggfunc="first"),
            run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        )

        self.evaluation_table = self.evaluation_table.reset_index()
        self.evaluation_table = self.evaluation_table[['quantifier', 'dataset', 'sample_size', 'sampling_seed', 'alpha', 'pred_prev', 'inv_abs_error', 'run_time']]
        self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset"]).agg(
            inv_abs_error = pd.NamedAgg(column="inv_abs_error", aggfunc="mean"),
            run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        )

        self.evaluation_table.to_csv("eval_table.csv")
        

    def recommend(self, X):
        pass