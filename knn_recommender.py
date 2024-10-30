from base_recommender import BaseRecommender

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

    def recommend(self, X):
        pass

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

        X_train = self.meta_features_table.values
        # for quantifier in self.evaluation_table.index.levels[0].tolist():
        #     y_train = self.evaluation_table.loc[quantifier]['abs_error'].values
        #     self.recommender_dict[quantifier] = clone(self._recommender_model)
        #     self.recommender_dict[quantifier].fit(X_train, y_train)
        self._fitted = True