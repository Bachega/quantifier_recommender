import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import is_regressor, clone

from meta_feature_extractor import MetaFeatureExtractor
from quantifier_evaluator import QuantifierEvaluator
    
class QuantifierRecommender:
    def __init__(self, supervised: bool = True, recommender_model = RandomForestRegressor()):
        if not is_regressor(recommender_model):
            raise ValueError("The regression model must be a scikit-learn regressor")
        self.__recommender_model = recommender_model
        
        self.__supervised = supervised
        self.is_supervised = supervised

        self.__unscaled_meta_features_table = None
        self.meta_features_table = None

        self.evaluation_table = None
        
        self.recommender_dict = {}
        self.is_meta_table_constructed = False


        self.mfe = MetaFeatureExtractor()
        self.quantifier_evaluator = QuantifierEvaluator()
     
    def __get_normalized_meta_features_table(self):
        columns = self.__unscaled_meta_features_table.columns
        data = self.__unscaled_meta_features_table.values

        self._fitted_scaler = MinMaxScaler()
        self._fitted_scaler.fit(data)

        scaled_meta_features_table = pd.DataFrame(self._fitted_scaler.transform(data), columns=columns)
        scaled_meta_features_table.index = self.__unscaled_meta_features_table.index
        return scaled_meta_features_table
    
    def __extract_and_append(self, dataset_name, X, y = None):
        columns, features = self.mfe.extract_meta_features(X, y)

        if self.__unscaled_meta_features_table is None:
            self.__unscaled_meta_features_table = pd.DataFrame(columns=columns)

        self.__unscaled_meta_features_table.loc[dataset_name] = features
    
    def __load_train_test_data(self, dataset_name, train_data_path, test_data_path):
        train_df = pd.read_csv(f"{train_data_path}/{dataset_name}.csv")
        y_train = train_df.pop(train_df.columns[-1])
        X_train = train_df

        test_df = pd.read_csv(f"{test_data_path}/{dataset_name}.csv")
        y_test = test_df.pop(test_df.columns[-1])
        X_test = test_df

        return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    def save_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        with pd.HDFStore(meta_table_path) as store:
            store.put("meta_features_table", self.meta_features_table)
            store.put("unscaled_meta_features_table", self.__unscaled_meta_features_table)
            store.put("not_agg_evaluation_table", self.__not_agg_evaluation_table)
            store.put("evaluation_table", self.evaluation_table)
    
    def load_fit_meta_table(self, meta_table_path: str):
        if not meta_table_path.endswith(".h5"):
            meta_table_path += ".h5"

        with pd.HDFStore(meta_table_path) as store:
            self.meta_features_table = store.get("meta_features_table")
            self.__unscaled_meta_features_table = store.get("unscaled_meta_features_table")
            self.__not_agg_evaluation_table = store.get("not_agg_evaluation_table")
            self.evaluation_table = store.get("evaluation_table")

        data = self.__unscaled_meta_features_table.values
        self._fitted_scaler = MinMaxScaler()
        self._fitted_scaler.fit(data)

        X_train = self.meta_features_table.values
        for quantifier in self.evaluation_table.index.levels[0].tolist():
            y_train = self.evaluation_table.loc[quantifier]['abs_error'].values
            self.recommender_dict[quantifier] = clone(self.__recommender_model)
            self.recommender_dict[quantifier].fit(X_train, y_train)
    
    def persist_model(self, path: str):
        with open(path, "wb") as file_handler:
            joblib.dump(self, file_handler)
    
    @staticmethod
    def load_model(path: str):
        quantifier_recommender = None
        with open(path, "rb") as file_handler:
            quantifier_recommender = joblib.load(file_handler)
        
        return quantifier_recommender

    def fit(self, datasets_path: str, train_data_path: str, test_data_path: str):
        dataset_list = [csv for csv in os.listdir(datasets_path) if csv.endswith(".csv")]

        # Appending the evaluations to a list and then concatenating them
        # to a pandas dataframe is O(n)
        evaluation_list = []
        for i, dataset in enumerate(dataset_list):
            dataset_name = dataset.split(".csv")[0]
            
            # Meta-Features extraction
            dt = pd.read_csv(datasets_path + dataset)
            dt = dt.dropna()
            
            if self.__supervised:
                y = dt.pop('class')
            else:
                y = None
            X = dt
            
            self.__extract_and_append(dataset_name, X=X, y=y)

            # Quantifiers evaluation
            X_train, y_train, X_test, y_test = self.__load_train_test_data(dataset_name, train_data_path, test_data_path)
            evaluation_list.append(self.quantifier_evaluator.evaluate_quantifiers(dataset_name,
                                                                                    X_train,
                                                                                    y_train,
                                                                                    X_test,
                                                                                    y_test))
            # # DELETE THIS
            # if i == 5:
            #     break
        # # DELETE THIS
        # eval_table = pd.concat(evaluation_list, axis=0)
        # eval_table.to_csv("./eval_table.csv", index=False)
        # return eval_table

        # Normalize the extracted meta-features
        self.meta_features_table = self.__get_normalized_meta_features_table()

        # Concatenate all the evaluations into a single evaluation table
        # and then sort and aggregate the quantifiers evaluations
        self.__not_agg_evaluation_table = pd.concat(evaluation_list, axis=0)

        self.evaluation_table = self.__not_agg_evaluation_table.sort_values(by=['quantifier', 'dataset'])
        self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset", "alpha"]).agg(
            pred_prev = pd.NamedAgg(column="pred_prev", aggfunc="mean"),
            abs_error = pd.NamedAgg(column="abs_error", aggfunc="mean"),
            sample_size = pd.NamedAgg(column="sample_size", aggfunc="first"),
            sampling_seed = pd.NamedAgg(column="sampling_seed", aggfunc="first"),
            run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        )

        self.evaluation_table = self.evaluation_table.reset_index()
        self.evaluation_table = self.evaluation_table[['quantifier', 'dataset', 'sample_size', 'sampling_seed', 'alpha', 'pred_prev', 'abs_error', 'run_time']]
        self.evaluation_table = self.evaluation_table.groupby(["quantifier", "dataset"]).agg(
            abs_error = pd.NamedAgg(column="abs_error", aggfunc="mean"),
            run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        )
        
        X_train = self.meta_features_table.values
        y_train = None
        for quantifier in self.evaluation_table.index.levels[0].tolist():
            y_train = self.evaluation_table.loc[quantifier]['abs_error'].values
            self.recommender_dict[quantifier] = clone(self.__recommender_model)
            self.recommender_dict[quantifier].fit(X_train, y_train)
    
    def predict(self, X_test, y_test = None, k: int = -1) -> dict:
        if self.__supervised:
            _, _X_test = self.mfe.extract_meta_features(X_test, y_test)
        else:
            _, _X_test = self.mfe.extract_meta_features(X_test)
        _X_test = self._fitted_scaler.fit_transform(np.array(_X_test).reshape(1, -1))

        if k == -1 or k > len(self.recommender_dict.keys()):
            k = len(self.recommender_dict.keys())
    
        result = pd.Series(index = list(self.recommender_dict.keys()))
        for quantifier, recommender in self.recommender_dict.items():
            result[quantifier] = recommender.predict(_X_test)

        ranking = {key: None for key in range(1, k+1)}        
        i = 1
        for index, value in result.sort_values().items():
            ranking[i] = [index, value]
            if i == k:
                break
            i += 1
        
        return ranking

    # Evaluate Quantifier Recommender with Leave-One-Out
    def leave_one_out_evaluation(self, recommender_eval_path: str = None, quantifiers_eval_path: str = None):
        aux_recommender_evaluation_table = pd.DataFrame(columns=["predicted_error", "true_error"], index=self.evaluation_table.index)
        for quantifier, recommender in self.recommender_dict.items():
            recommender_ = clone(recommender)
            scaler = MinMaxScaler()

            for dataset in self.evaluation_table.index.levels[1]:
                unscaled_X_test = self.__unscaled_meta_features_table.loc[dataset].values
                y_test = self.evaluation_table.loc[quantifier, dataset]['abs_error']

                unscaled_X_train = self.__unscaled_meta_features_table.drop(index=dataset).values
                y_train = self.evaluation_table.loc[quantifier].drop(index=dataset)['abs_error'].values

                scaler.fit(unscaled_X_train)
                X_train = scaler.transform(unscaled_X_train)
                recommender_.fit(X_train, y_train)

                X_test = scaler.fit_transform(np.array(unscaled_X_test).reshape(1, -1))
                predicted_error = recommender_.predict(X_test)[0]

                aux_recommender_evaluation_table.loc[(quantifier, dataset)] = [predicted_error, y_test]
        
        datasets = aux_recommender_evaluation_table.index.get_level_values('dataset').unique()
        recommender_evaluation_table = pd.DataFrame(columns=["predicted_ranking", "true_ranking", "predicted_ranking_error", "true_ranking_error"], index=datasets)
        for dataset in datasets:
            filtered_result = aux_recommender_evaluation_table.xs(dataset, level='dataset')
            
            predicted_ranking = filtered_result.sort_values(by='predicted_error').index.tolist()
            predicted_ranking_error = [filtered_result.loc[quantifier, 'predicted_error'] for quantifier in predicted_ranking]

            true_ranking = filtered_result.sort_values(by='true_error').index.tolist()
            true_ranking_error = [filtered_result.loc[quantifier, 'true_error'] for quantifier in true_ranking]

            recommender_evaluation_table.loc[dataset] = [predicted_ranking, true_ranking, predicted_ranking_error, true_ranking_error]
          
        if not recommender_eval_path is None:
            recommender_evaluation_table.to_csv(recommender_eval_path)
        
        if not quantifiers_eval_path is None:
            self.__not_agg_evaluation_table.to_csv(quantifiers_eval_path)
        
        not_agg_evaluation_table = self.__not_agg_evaluation_table.copy(deep=True)
        not_agg_evaluation_table.sort_values(by=['quantifier', 'dataset'], inplace=True)
        not_agg_evaluation_table.reset_index(drop=True, inplace=True)
        
        return recommender_evaluation_table, not_agg_evaluation_table
    
    def get_not_agg(self):
        return self.__not_agg_evaluation_table