import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import is_regressor, clone

from base_recommender import BaseRecommender

class RegressionRecommender(BaseRecommender):
    def __init__(self, supervised: bool = True, model = RandomForestRegressor(n_jobs=-1), _load: bool = True):
        assert is_regressor(model), "The regression model must be a scikit-learn regressor"

        super().__init__(supervised=supervised, _load=_load)
        if _load == False:
            self.model = model
            self.model_dict = {}
            self._unscaled_meta_features_table = None
            self.meta_features_table = None
            self.evaluation_table = None
    
    def save_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        with pd.HDFStore(meta_table_path) as store:
            store.put("meta_features_table", self.meta_features_table)
            store.put("unscaled_meta_features_table", self._unscaled_meta_features_table)
            store.put("not_agg_evaluation_table", self._not_agg_evaluation_table)
            store.put("evaluation_table", self.evaluation_table)
            store.put('scaler_method', pd.Series([self._scaler_method]), format="table")
    
    def load_fit_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        with pd.HDFStore(meta_table_path) as store:
            self.meta_features_table = store.get("meta_features_table")
            self._unscaled_meta_features_table = store.get("unscaled_meta_features_table")
            self._not_agg_evaluation_table = store.get("not_agg_evaluation_table")
            self.evaluation_table = store.get("evaluation_table")
            self._scaler_method = store.get('scaler_method').values[0]

        data = self._unscaled_meta_features_table.values
        if self._scaler_method == "zscore":
            self._fitted_scaler = StandardScaler()
        elif self._scaler_method == "minmax":
            self._fitted_scaler = MinMaxScaler()
        self._fitted_scaler.fit(data)

        X_train = self.meta_features_table.values
        for quantifier in self.evaluation_table.index.levels[0].tolist():
            y_train = self.evaluation_table.loc[quantifier]['abs_error'].values
            self.model_dict[quantifier] = clone(self.model)
            self.model_dict[quantifier].fit(X_train, y_train)
        self._fitted = True
    
    def fit(self, full_set_path: str, train_set_path: str, test_set_path: str) -> None:
        dataset_list = [csv for csv in os.listdir(full_set_path) if csv.endswith(".csv")]

        # Appending the evaluations to a list and then concatenating them
        # to a pandas dataframe is O(n)
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
            evaluation_list.append(self.quantifier_evaluator.evaluate_quantifiers(dataset_name,
                                                                                    X_train,
                                                                                    y_train,
                                                                                    X_test,
                                                                                    y_test,
                                                                                    func_type="cost"))
            # DELETE THIS
            if i == 5:
                break

        # Normalize the extracted meta-features
        self.meta_features_table = self._get_normalized_meta_features_table(self._unscaled_meta_features_table)

        # Concatenate all the evaluations into a single evaluation table
        # and then sort and aggregate the quantifiers evaluations
        self._not_agg_evaluation_table = pd.concat(evaluation_list, axis=0)

        self.evaluation_table = self._not_agg_evaluation_table.sort_values(by=['quantifier', 'dataset'])
        self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset"]).agg(
            abs_error = pd.NamedAgg(column="abs_error", aggfunc="mean"),
            run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        )

        # self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset", "alpha"]).agg(
        #     pred_prev = pd.NamedAgg(column="pred_prev", aggfunc="mean"),
        #     abs_error = pd.NamedAgg(column="abs_error", aggfunc="mean"),
        #     sample_size = pd.NamedAgg(column="sample_size", aggfunc="first"),
        #     sampling_seed = pd.NamedAgg(column="sampling_seed", aggfunc="first"),
        #     run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        # )

        # self.evaluation_table = self.evaluation_table.reset_index()
        # self.evaluation_table = self.evaluation_table[['quantifier', 'dataset', 'sample_size', 'sampling_seed', 'alpha', 'pred_prev', 'abs_error', 'run_time']]
        # self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset"]).agg(
        #     abs_error = pd.NamedAgg(column="abs_error", aggfunc="mean"),
        #     run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        # )
        
        X_train = self.meta_features_table.values
        y_train = None
        for quantifier in self.evaluation_table.index.levels[0].tolist():
            y_train = self.evaluation_table.loc[quantifier]['abs_error'].values
            self.model_dict[quantifier] = clone(self.model)
            self.model_dict[quantifier].fit(X_train, y_train)
        self._fitted = True
    
    def recommend(self, X, y = None, k: int  = -1):
        assert self._fitted, "The model must be fitted before making predictions."
        assert k > 0 or k == -1, "The number of quantifiers to recommend must be greater than 0 or -1 to recommend all quantifiers."
        
        k = len(self.model_dict) if k == -1 else k

        if self.supervised:
            _, X_test = self.mfe.extract_meta_features(X, y)
        else:
            _, X_test = self.mfe.extract_meta_features(X)
        X_test = self._fitted_scaler.transform(np.array(X_test).reshape(1, -1))

        result = []
        i = 0
        for quantifier, recommender in self.model_dict.items():
            result.append((quantifier, recommender.predict(X_test)[0]))
            i += 1
            if i == k:
                break
            # result[quantifier] = recommender.predict(_X_test)

        quantifier_mae_pairs = sorted(result, key=lambda x: x[1], reverse=False)
        quantifiers, maes = zip(*quantifier_mae_pairs)
        errors = np.array(maes)
        denominator = np.sum(1/errors)
        weights = (1/errors)/denominator

        return tuple(quantifiers), tuple(weights)


    # Evaluate Quantifier Recommender with Leave-One-Out
    def leave_one_out_evaluation(self, recommender_eval_path: str = None, quantifiers_eval_path: str = None):
        assert self._fitted, "The model must be fitted before running the leave-one-out evaluation."

        aux_recommender_evaluation_table = pd.DataFrame(columns=["predicted_error", "true_error"], index=self.evaluation_table.index)
        for quantifier, recommender in self.model_dict.items():
            recommender_ = clone(recommender)
            if self._scaler_method == "zscore":
                scaler = StandardScaler()
            elif self._scaler_method == "minmax":
                scaler = MinMaxScaler()

            for dataset in self.evaluation_table.index.levels[1]:
                unscaled_X_test = self._unscaled_meta_features_table.loc[dataset].values
                y_test = self.evaluation_table.loc[quantifier, dataset]['abs_error']

                unscaled_X_train = self._unscaled_meta_features_table.drop(index=dataset).values
                y_train = self.evaluation_table.loc[quantifier].drop(index=dataset)['abs_error'].values

                scaler.fit(unscaled_X_train)
                X_train = scaler.transform(unscaled_X_train)
                recommender_.fit(X_train, y_train)

                X_test = scaler.transform(np.array(unscaled_X_test).reshape(1, -1))
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