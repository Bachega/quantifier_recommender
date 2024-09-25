import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from meta_feature_extractor import MetaFeatureExtractor
from quantifier_evaluator import QuantifierEvaluator
from utils_ import load_train_test_data
    
class QuantifierRecommender:
    def __init__(self, supervised: bool = True):
        self.__supervised = supervised

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
    
    def load_meta_features_table(self, meta_table_path):
        self.meta_features_table.read_csv(meta_table_path)
    
    def save_meta_features_table(self, meta_table_path: str = "./recommender_data/meta-table.csv"):
        self.meta_features_table.to_csv(meta_table_path)

    def load_evaluation_table(self, evaluation_table_path):
        self.evaluation_table.read_csv(evaluation_table_path)
    
    def save_evaluation_table(self, evaluation_table_path: str = "./recommender_data/evaluation-table.csv"):
        self.evaluation_table.to_csv(evaluation_table_path)

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
            if i == 2:
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
            self.recommender_dict[quantifier] = RandomForestRegressor()
            self.recommender_dict[quantifier].fit(X_train, y_train)
    
    def predict(self, X_test, y_test = None) -> dict:
        if self.__supervised:
            _, _X_test = self.mfe.extract_meta_features(X_test, y_test)
        else:
            _, _X_test = self.mfe.extract_meta_features(X_test)

        _X_test = self._fitted_scaler.fit_transform(np.array(_X_test).reshape(1, -1))

        result = pd.Series(index = list(self.recommender_dict.keys()))
        for quantifier, recommender in self.recommender_dict.items():
            result[quantifier] = recommender.predict(_X_test)

        ranking = {key: None for key in range(1, len(self.recommender_dict.keys())+1)}
        i = 1
        for index, value in result.sort_values().items():
            ranking[i] = [index, value]
            i += 1
        
        return ranking

    # Evaluate Quantifier Recommender with Leave-One-Out
    # def _leave_one_out(self):
