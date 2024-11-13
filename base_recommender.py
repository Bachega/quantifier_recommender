from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from meta_feature_extractor import MetaFeatureExtractor
from quantifier_evaluator import QuantifierEvaluator

class BaseRecommender(ABC):
    def __init__(self, supervised = True, _load: bool = True):
        if _load:
            self.__dict__.update(BaseRecommender.load_model(f"./recommender_data/{self.__class__.__name__}.joblib").__dict__)
        else:
            self.mfe = MetaFeatureExtractor()
            self.quantifier_evaluator = QuantifierEvaluator()
            self.supervised = supervised
            self._scaler_method = None

    @abstractmethod
    def fit(self, complete_data_path: str, train_data_path: str, test_data_path: str):
        pass

    @abstractmethod
    def recommend(self, X, y, k):
        pass
    
    def _extract_and_append(self, dataset_name: str, X, y = None, meta_features_table = None) -> None:
        columns, features = self.mfe.extract_meta_features(X, y)

        if meta_features_table is None:
            meta_features_table = pd.DataFrame(columns=columns)

        meta_features_table.loc[dataset_name] = features
        return meta_features_table

    def _load_train_test_set(self, dataset_name: str, train_set_path: str, test_set_path: str):
        train_df = pd.read_csv(f"{train_set_path}/{dataset_name}.csv")
        y_train = train_df.pop(train_df.columns[-1])
        X_train = train_df

        test_df = pd.read_csv(f"{test_set_path}/{dataset_name}.csv")
        y_test = test_df.pop(test_df.columns[-1])
        X_test = test_df

        return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    def persist_model(self, path: str = None):
        if not path:
            path = f"{self.__class__.__name__}.joblib"
        with open(path, "wb") as file_handler:
            joblib.dump(self, file_handler)
    
    @staticmethod
    def load_model(path: str):
        quantifier_recommender = None
        with open(path, "rb") as file_handler:
            quantifier_recommender = joblib.load(file_handler)
        return quantifier_recommender