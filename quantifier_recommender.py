import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import is_regressor, clone

from meta_feature_extractor import MetaFeatureExtractor
from quantifier_evaluator import QuantifierEvaluator
from utils_ import load_train_test_data
    
class QuantifierRecommender:
    def __init__(self, supervised: bool = True, recommender_model = RandomForestRegressor()):
        if not is_regressor(recommender_model):
            raise ValueError("The regression model must be a scikit-learn regressor")
        self.__recommender_model = recommender_model
        
        self.__supervised = supervised
        self.is_supervised = supervised

        self._unscaled_meta_features_table = None
        self.meta_features_table = None

        self.evaluation_table = None
        
        self.recommender_dict = {}
        self.is_meta_table_constructed = False


        self.mfe = MetaFeatureExtractor()
        self.quantifier_evaluator = QuantifierEvaluator()
     
    def __get_normalized_meta_features_table(self):
        columns = self._unscaled_meta_features_table.columns
        data = self._unscaled_meta_features_table.values

        self._fitted_scaler = MinMaxScaler()
        self._fitted_scaler.fit(data)

        scaled_meta_features_table = pd.DataFrame(self._fitted_scaler.transform(data), columns=columns)
        scaled_meta_features_table.index = self._unscaled_meta_features_table.index
        return scaled_meta_features_table
    
    def __extract_and_append(self, dataset_name, X, y = None):
        columns, features = self.mfe.extract_meta_features(X, y)

        if self._unscaled_meta_features_table is None:
            self._unscaled_meta_features_table = pd.DataFrame(columns=columns)

        # self.meta_features_table.loc[len(self.meta_features_table.index)] = features
        self._unscaled_meta_features_table.loc[dataset_name] = features
    
    def save_meta_table(self, meta_features_table_path: str, evaluation_table_path: str):
        if not ".csv" in meta_features_table_path:
            meta_features_table_path += ".csv"
        unscaled_meta_features_table_path = meta_features_table_path.replace(".csv", "_unscaled.csv")
        
        if not ".csv" in evaluation_table_path:
            evaluation_table_path += ".csv"

        self.meta_features_table.to_csv(meta_features_table_path)
        self._unscaled_meta_features_table.to_csv(unscaled_meta_features_table_path)
        self.evaluation_table.to_csv(evaluation_table_path)
    
    # def load_meta_table(self, meta_features_table_path: str, evaluation_table_path: str):        
    #     if not ".csv" in meta_features_table_path:
    #         unscaled_meta_features_table_path += "_unscaled.csv"
    #         meta_features_table_path += ".csv"
    #     unscaled_meta_features_table_path = meta_features_table_path.replace(".csv", "_unscaled.csv")
        
    #     if not ".csv" in evaluation_table_path:
    #         evaluation_table_path += ".csv"
        
    #     self.meta_features_table = pd.read_csv(meta_features_table_path, index_col=0)
    #     self._unscaled_meta_features_table = pd.read_csv(unscaled_meta_features_table_path, index_col=0)
    #     self.evaluation_table = pd.read_csv(evaluation_table_path, index_col=[0, 1])

    def load_and_fit_meta_table(self, meta_features_table_path: str, evaluation_table_path: str):
        # self.load_meta_table(meta_fetaures_table_path, evaluation_table_path)
        if not ".csv" in meta_features_table_path:
            unscaled_meta_features_table_path += "_unscaled.csv"
            meta_features_table_path += ".csv"
        unscaled_meta_features_table_path = meta_features_table_path.replace(".csv", "_unscaled.csv")
        
        if not ".csv" in evaluation_table_path:
            evaluation_table_path += ".csv"
        
        self.meta_features_table = pd.read_csv(meta_features_table_path, index_col=0)
        self._unscaled_meta_features_table = pd.read_csv(unscaled_meta_features_table_path, index_col=0)
        self.evaluation_table = pd.read_csv(evaluation_table_path, index_col=[0, 1])

        data = self._unscaled_meta_features_table.values
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
            X_train, y_train, X_test, y_test = load_train_test_data(dataset_name, train_data_path, test_data_path)
            evaluation_list.append(self.quantifier_evaluator.evaluate_quantifiers(dataset_name,
                                                                                    X_train,
                                                                                    y_train,
                                                                                    X_test,
                                                                                    y_test))
            if i == 4:
                break
            
        # Normalize the extracted meta-features
        self.meta_features_table = self.__get_normalized_meta_features_table()

        # Concatenate all the evaluations into a single evaluation table
        # and then sort and aggregate the quantifiers evaluations
        self.evaluation_table = pd.concat(evaluation_list, axis=0)
        self.evaluation_table.sort_values(by=['quantifier', 'dataset'], inplace=True)
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
    def leave_one_out_evaluation(self, path: str = None):
        recommender_evaluation_table = pd.DataFrame(columns=["predicted_error", "true_error"], index=self.evaluation_table.index)
        for quantifier, recommender in self.recommender_dict.items():
            recommender_ = clone(recommender)
            scaler = MinMaxScaler()

            for dataset in self.evaluation_table.index.levels[1]:
                unscaled_X_test = self._unscaled_meta_features_table.loc[dataset].values
                y_test = self.evaluation_table.loc[quantifier, dataset]['abs_error']

                unscaled_X_train = self._unscaled_meta_features_table.drop(index=dataset).values
                y_train = self.evaluation_table.loc[quantifier].drop(index=dataset)['abs_error'].values

                scaler.fit(unscaled_X_train)
                X_train = scaler.transform(unscaled_X_train)
                recommender_.fit(X_train, y_train)

                X_test = scaler.fit_transform(np.array(unscaled_X_test).reshape(1, -1))
                predicted_error = recommender_.predict(X_test)[0]

                recommender_evaluation_table.loc[(quantifier, dataset)] = [predicted_error, y_test]
        
        if not path is None:
            recommender_evaluation_table.to_csv(path)
        return recommender_evaluation_table
    
    def OLD_leave_one_out_evaluation(self, path: str = None):
        recommender_evaluation_table = pd.DataFrame(columns=["predicted_error", "true_error"], index=self.evaluation_table.index)
        for quantifier, recommender in self.recommender_dict.items():
            recommender_ = clone(recommender)

            for dataset in self.evaluation_table.index.levels[1]:
                X_test = self.meta_features_table.loc[dataset].values
                y_test = self.evaluation_table.loc[quantifier, dataset]['abs_error']

                X_train = self.meta_features_table.drop(index=dataset).values
                y_train = self.evaluation_table.loc[quantifier].drop(index=dataset)['abs_error'].values

                recommender_.fit(X_train, y_train)
                predicted_error = recommender_.predict(np.array(X_test).reshape(1, -1))[0]

                recommender_evaluation_table.loc[(quantifier, dataset)] = [predicted_error, y_test]
        
        if not path is None:
            recommender_evaluation_table.to_csv(path)
        return recommender_evaluation_table
