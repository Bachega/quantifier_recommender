from .base_recommender import BaseRecommender

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import is_regressor, clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

class RegressionRecommender(BaseRecommender):
    def __init__(self, supervised: bool = True, model = RandomForestRegressor(n_jobs=-1), load_default: bool = True):
        assert is_regressor(model), "The regression model must be a scikit-learn regressor"

        if load_default == False:
            self.__model = Pipeline([
                ("normalization", StandardScaler()),
                ("variance_threshold", VarianceThreshold()),
                ("model", model)
            ])
            self.model_dict = {}

            self.meta_features_table = None
            self._not_agg_evaluation_table = None
            self.evaluation_table = None
        super().__init__(supervised=supervised, load_default=load_default)
            
    def save_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        scaler = self.__model.named_steps['normalization']
        if isinstance(scaler, MinMaxScaler):
            scaler_method = "minmax"
        elif isinstance(scaler, StandardScaler):
            scaler_method = "zscore"

        with pd.HDFStore(meta_table_path) as store:
            store.put("meta_features_table", self.meta_features_table)
            store.put("not_agg_evaluation_table", self._not_agg_evaluation_table)
            store.put("evaluation_table", self.evaluation_table)
            store.put("scaler_method", pd.Series([scaler_method], index=["scaler_method"]))
    
    def load_fit_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        with pd.HDFStore(meta_table_path) as store:
            self.meta_features_table = store.get("meta_features_table")
            self._not_agg_evaluation_table = store.get("not_agg_evaluation_table")
            self.evaluation_table = store.get("evaluation_table")
            self._scaler_method = store["scaler_method"].iloc[0]
        
        if self._scaler_method == "minmax":
            scaler = MinMaxScaler()
        elif self._scaler_method == "zscore":
            scaler = StandardScaler()
        
        model = type(self.__model.named_steps['model'])()
        self.__model = Pipeline([
            ("normalization", scaler),
            ("variance_threshold", VarianceThreshold()),
            ("model", model)
        ])

        X_train = self.meta_features_table.values
        for quantifier in self.evaluation_table.index.levels[0].tolist():
            y_train = self.evaluation_table.loc[quantifier]['abs_error'].values
            self.model_dict[quantifier] = clone(self.__model)
            self.model_dict[quantifier].fit(X_train, y_train)
        self._fitted = True
    
    def _fit_method(self, meta_features_table, not_aggregated_evaluation_table, evaluation_table) -> 'RegressionRecommender':
        self.meta_features_table = meta_features_table
        self._not_agg_evaluation_table = not_aggregated_evaluation_table
        self.evaluation_table = evaluation_table
        
        X_train = self.meta_features_table.values
        y_train = None
        for quantifier in self.evaluation_table.index.levels[0].tolist():
            y_train = self.evaluation_table.loc[quantifier]['abs_error'].values
            self.model_dict[quantifier] = clone(self.__model)
            self.model_dict[quantifier].fit(X_train, y_train)
        self._fitted = True
        return self
    
    def recommend(self, X, y = None, k: int  = -1):
        if self._load_default:
            self.load_model()

        assert self._fitted, "The model must be fitted before making predictions."
        assert k > 0 or k == -1, "The number of quantifiers to recommend must be greater than 0 or -1 to recommend all quantifiers."
        
        k = len(self.model_dict) if k == -1 else k
        
        _, X_test = self.mfe.extract_meta_features(X, y)
        X_test = np.array(X_test).reshape(1, -1)
        
        result = []
        i = 0
        for quantifier, recommender in self.model_dict.items():
            result.append((quantifier, recommender.predict(X_test)[0]))
            i += 1
            if i == k:
                break

        quantifier_mae_pairs = sorted(result, key=lambda x: x[1], reverse=False)
        quantifiers, maes = zip(*quantifier_mae_pairs)
        errors = np.array(maes)
        if np.any(errors == 0):
            errors = np.array([1e-6 if x == 0 else x for x in errors])
        denominator = np.sum(1/errors)
        weights = (1/errors)/denominator

        return tuple(quantifiers), tuple(weights)


    # Evaluate Quantifier Recommender with Leave-One-Out
    def leave_one_out_evaluation(self, recommender_eval_path: str = None, quantifiers_eval_path: str = None):
        assert self._fitted, "The model must be fitted before running the leave-one-out evaluation."

        aux_recommender_evaluation_table = pd.DataFrame(columns=["predicted_error", "true_error"], index=self.evaluation_table.index)
        for quantifier, recommender in self.model_dict.items():
            recommender_ = clone(recommender)
            for dataset in self.evaluation_table.index.levels[1]:
                X_test = self.meta_features_table.loc[dataset].values
                X_test = np.array(X_test).reshape(1, -1)
                y_test = self.evaluation_table.loc[quantifier, dataset]['abs_error']

                X_train = self.meta_features_table.drop(index=dataset).values
                y_train = self.evaluation_table.loc[quantifier].drop(index=dataset)['abs_error'].values

                recommender_.fit(X_train, y_train)
                predicted_error = recommender_.predict(X_test)[0]

                aux_recommender_evaluation_table.loc[(quantifier, dataset)] = [predicted_error, y_test]
        
        datasets = aux_recommender_evaluation_table.index.get_level_values('dataset').unique()
        recommender_evaluation_table = pd.DataFrame(columns=["predicted_ranking", "predicted_ranking_weights", "predicted_ranking_mae",
                                                             "true_ranking", "true_ranking_weights", "true_ranking_mae"], index=datasets)
        for dataset in datasets:
            filtered_result = aux_recommender_evaluation_table.xs(dataset, level='dataset')
            
            predicted_ranking = filtered_result.sort_values(by='predicted_error').index.tolist()
            predicted_ranking_mae = [filtered_result.loc[quantifier, 'predicted_error'] for quantifier in predicted_ranking]

            errors = np.array(predicted_ranking_mae)
            denominator = np.sum(1/errors)
            predicted_ranking_weights = (1/errors)/denominator

            true_ranking = filtered_result.sort_values(by='true_error').index.tolist()
            true_ranking_mae = [filtered_result.loc[quantifier, 'true_error'] for quantifier in true_ranking]

            errors = np.array(true_ranking_mae)
            if np.any(errors == 0):
                errors = np.array([1e-6 if x == 0 else x for x in errors])
            denominator = np.sum(1/errors)
            true_ranking_weights = (1/errors)/denominator

            recommender_evaluation_table.loc[dataset] = [predicted_ranking, predicted_ranking_weights, predicted_ranking_mae,
                                                         true_ranking, true_ranking_weights, true_ranking_mae]
          
        if not recommender_eval_path is None:
            recommender_evaluation_table.to_csv(recommender_eval_path)
        
        if not quantifiers_eval_path is None:
            self._not_agg_evaluation_table.to_csv(quantifiers_eval_path)
        
        not_agg_evaluation_table = self._not_agg_evaluation_table.copy(deep=True)
        not_agg_evaluation_table.sort_values(by=['quantifier', 'dataset'], inplace=True)
        not_agg_evaluation_table.reset_index(drop=True, inplace=True)
        
        return recommender_evaluation_table, not_agg_evaluation_table