from base_recommender import BaseRecommender

import os
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors

import pdb

class KNNRecommender(BaseRecommender):
    def __init__(self, supervised: bool = True, n_neighbors: int = 1, _load: bool = True):
        super().__init__(supervised, _load)
        if _load == False:
            self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan', n_jobs=-1)
            self._unscaled_meta_features_table = None
            self.meta_features_table = None
            self.arr_table = None
            self.n_neighbors = n_neighbors
    
    @property
    def n_neighbors(self):
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n_neighbors: int):
        assert n_neighbors > 0 or n_neighbors == -1, "The number of neighbors must be greater than 0 or -1 (all neighbors)."
        self._n_neighbors = n_neighbors
            
    def fit(self, full_set_path: str, train_set_path: str, test_set_path: str) -> None:
        dataset_list = [csv for csv in os.listdir(full_set_path) if csv.endswith(".csv")]
        evaluation_list = []
        for i, dataset in enumerate(dataset_list):
            dataset_name = dataset.split(".csv")[0]
            
            # Meta-Features extraction
            dt = pd.read_csv(full_set_path + dataset)
            dt = dt.dropna()
            
            if self.supervised:
                y = dt.pop('class')
            else:
                y = None
            X = dt
            
            self._unscaled_meta_features_table = self._extract_and_append(dataset_name, X, y, self._unscaled_meta_features_table)

            # Quantifiers evaluation
            X_train, y_train, X_test, y_test = self._load_train_test_set(dataset_name, train_set_path, test_set_path)
            evaluation_list.append(self.quantifier_evaluator.evaluate_quantifiers(dataset_name=dataset_name,
                                                                                  X_train=X_train,
                                                                                  y_train=y_train,
                                                                                  X_test=X_test,
                                                                                  y_test=y_test,
                                                                                  func_type="utility"))
            # DELETE THIS
            if i == 5:
                break

        # Normalize the extracted meta-features
        self.meta_features_table = self._get_normalized_meta_features_table(self._unscaled_meta_features_table, method="minmax")

        # Concatenate all the evaluations into a single evaluation table
        # and then sort and aggregate the quantifiers evaluations
        self._not_agg_evaluation_table = pd.concat(evaluation_list, axis=0)

        self.evaluation_table = self._not_agg_evaluation_table.sort_values(by=['quantifier', 'dataset'])
        self.evaluation_table = self.evaluation_table.sort_values(by=['quantifier', 'dataset'])
        self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset"]).agg(
            inv_abs_error = pd.NamedAgg(column="inv_abs_error", aggfunc="mean"),
            run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        )
        eval_table = self.evaluation_table.reset_index()
        
        dt_list = eval_table['dataset'].unique().tolist()
        qtf_list = eval_table['quantifier'].unique().tolist()
        self.arr_table = pd.DataFrame(columns=qtf_list)
        alpha = 0
        m = len(qtf_list) - 1
        for dt in dt_list:
            rows_by_dataset = eval_table[eval_table['dataset'] == dt]
            arr_row = []
            for qtf in qtf_list:
                acc_i = np.array(rows_by_dataset[rows_by_dataset['quantifier'] == qtf]['inv_abs_error'].values)
                acc_j = np.array(rows_by_dataset[rows_by_dataset['quantifier'] != qtf]['inv_abs_error'].values)

                run_time_i = np.array(rows_by_dataset[rows_by_dataset['quantifier'] == qtf]['run_time'].values)
                run_time_j = np.array(rows_by_dataset[rows_by_dataset['quantifier'] != qtf]['run_time'].values)

                acc_i_div_j = acc_i / acc_j
                run_time_i_div_j = 1 + (alpha * np.log10(run_time_i / run_time_j))
                arr_i = np.sum(acc_i_div_j / run_time_i_div_j) / m

                arr_row.append(arr_i)
            self.arr_table.loc[dt] = arr_row

        self.model.fit(self.meta_features_table.values)
        self._fitted = True  

    def recommend(self, X, y = None):
        assert self._fitted, "The model must be fitted before making predictions."

        if self.supervised:
            _, X_test = self.mfe.extract_meta_features(X, y)
        else:
            _, X_test = self.mfe.extract_meta_features(X)
        X_test = self._fitted_scaler.transform(np.array(X_test).reshape(1, -1))

        distances, indices = self.model.kneighbors(X_test.reshape(1, -1))
        distances, indices = distances[0], indices[0]

        quantifiers = self.arr_table.columns
        new_arr_array = np.array(len(quantifiers) * [np.float64(0)])
        weights = np.array(1/distances) / np.sum(1/distances)
        for idx, w in zip(indices, weights):
            arr_idx = self.meta_features_table.iloc[idx].name
            new_arr_array += np.array(self.arr_table.loc[arr_idx].values) * w

        # List of tuples: (quantifier, ARR value) sorted by their ARR value
        quantifier_arr_pairs = list(zip(quantifiers, new_arr_array))
        quantifier_arr_pairs = sorted(quantifier_arr_pairs, key=lambda x: x[1], reverse=True)
        quantifiers, arrs = zip(*quantifier_arr_pairs)

        # Calculate the weights of the quantifiers
        # by their ARR values (proportional)
        weights = np.array(arrs) / np.sum(arrs)

        # Return two tuples: (quantifiers), (weights)
        return tuple(quantifiers), tuple(weights)

    def save_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        with pd.HDFStore(meta_table_path) as store:
            store.put("meta_features_table", self.meta_features_table)
            store.put("unscaled_meta_features_table", self._unscaled_meta_features_table)
            store.put("not_agg_evaluation_table", self._not_agg_evaluation_table)
            store.put("evaluation_table", self.evaluation_table)
            store.put("arr_table", self.arr_table)
            store.put('scaler_method', pd.Series([self._scaler_method]), format="table")
    
    def load_fit_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        with pd.HDFStore(meta_table_path) as store:
            self.meta_features_table = store.get("meta_features_table")
            self._unscaled_meta_features_table = store.get("unscaled_meta_features_table")
            self._not_agg_evaluation_table = store.get("not_agg_evaluation_table")
            self.evaluation_table = store.get("evaluation_table")
            self.arr_table = store.get("arr_table")
            self._scaler_method = store.get('scaler_method').values[0]

        data = self._unscaled_meta_features_table.values
        if self._scaler_method == "zscore":
            self._fitted_scaler = StandardScaler()
        elif self._scaler_method == "minmax":
            self._fitted_scaler = MinMaxScaler()
        self._fitted_scaler.fit(data)

        self.model.fit(self.meta_features_table.values)
        self._fitted = True
    
    # Evaluate Quantifier Recommender with Leave-One-Out
    def leave_one_out_evaluation(self, recommender_eval_path: str = None, quantifiers_eval_path: str = None):
        assert self._fitted, "The model must be fitted before running the leave-one-out evaluation."
        predicted_arr_table = pd.DataFrame(columns=self.arr_table.columns, index=self.arr_table.index.tolist())
        true_arr_table = pd.DataFrame(columns=self.arr_table.columns, index=self.arr_table.index.tolist())
        recommender_evaluation_table = pd.DataFrame(columns=["predicted_ranking",
                                                             "predicted_ranking_weights",
                                                             "predicted_ranking_arr",
                                                             "true_ranking",
                                                             "true_ranking_weights",
                                                             "true_ranking_arr"], index=self.arr_table.index.tolist())

        recommender_ = clone(self.model)
        if self._scaler_method == "zscore":
            scaler = StandardScaler()
        elif self._scaler_method == "minmax":
            scaler = MinMaxScaler()

        for dataset in self.arr_table.index.tolist():
            unscaled_X_test = self._unscaled_meta_features_table.loc[dataset].values
            y_test = self.arr_table.loc[dataset].values

            unscaled_X_train = self._unscaled_meta_features_table.drop(index=dataset).values
            y_train = (self.arr_table.drop(index=dataset)).values

            X_train = scaler.fit_transform(unscaled_X_train)
            recommender_.fit(X_train, y_train)

            X_test = scaler.transform(np.array(unscaled_X_test).reshape(1, -1))
            distances, indices = recommender_.kneighbors(X_test.reshape(1, -1))
            distances, indices = distances[0], indices[0]
            quantifiers = self.arr_table.columns
            new_arr_array = np.array(len(quantifiers) * [np.float64(0)])
            weights = np.array(1/distances) / np.sum(1/distances)
            for idx, w in zip(indices, weights):
                arr_idx = self.meta_features_table.iloc[idx].name
                new_arr_array += np.array(self.arr_table.loc[arr_idx].values) * w

            quantifier_arr_pairs = sorted(list(zip(quantifiers, new_arr_array)), key=lambda x: x[1], reverse=True)
            predicted_ranking, predicted_arr = zip(*quantifier_arr_pairs)
            predicted_ranking_weights = np.array(predicted_arr) / np.sum(predicted_arr)

            quantifier_arr_pairs = sorted(list(zip(quantifiers, y_test)), key=lambda x: x[1], reverse=True)
            true_ranking, true_arr = zip(*quantifier_arr_pairs)
            true_ranking_weights = np.array(true_arr) / np.sum(true_arr)

            recommender_evaluation_table.loc[dataset] = [predicted_ranking, predicted_ranking_weights, predicted_arr,
                                                         true_ranking, true_ranking_weights, true_arr]
            
        if not recommender_eval_path is None:
            recommender_evaluation_table.to_csv(recommender_eval_path)
        
        if not quantifiers_eval_path is None:
            self._not_agg_evaluation_table.to_csv(quantifiers_eval_path)
        
        not_agg_evaluation_table = self._not_agg_evaluation_table.copy(deep=True)
        not_agg_evaluation_table.sort_values(by=['quantifier', 'dataset'], inplace=True)
        not_agg_evaluation_table.reset_index(drop=True, inplace=True)
        
        return recommender_evaluation_table, not_agg_evaluation_table