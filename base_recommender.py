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
            self._fitted = False
            self._fitted_scaler = None
            self._scaler_method = None

    @abstractmethod
    def fit(self, complete_data_path: str, train_data_path: str, test_data_path: str):
        pass

    @abstractmethod
    def recommend(self, X, y, k):
        pass
    
    def _get_normalized_meta_features_table(self, unscaled_meta_features_table: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
        assert isinstance(unscaled_meta_features_table, pd.DataFrame), "Invalid input. Input must be a pandas DataFrame."
        assert method in ["minmax", "zscore"], "Invalid normalization method. Choose between 'minmax' and 'zscore'."
        
        columns = unscaled_meta_features_table.columns
        data = unscaled_meta_features_table.values

        if method == "minmax":
            self._fitted_scaler = MinMaxScaler()
        elif method == "zscore":
            self._fitted_scaler = StandardScaler()
        self._scaler_method = method
        self._fitted_scaler.fit(data)

        scaled_meta_features_table = pd.DataFrame(self._fitted_scaler.transform(data), columns=columns)
        scaled_meta_features_table.index = unscaled_meta_features_table.index
        return scaled_meta_features_table

    def _extract_and_append(self, dataset_name: str, X, y = None, unscaled_meta_features_table = None) -> None:
        columns, features = self.mfe.extract_meta_features(X, y)

        if unscaled_meta_features_table is None:
            unscaled_meta_features_table = pd.DataFrame(columns=columns)

        unscaled_meta_features_table.loc[dataset_name] = features
        return unscaled_meta_features_table

    def _load_train_test_data(self, dataset_name: str, train_data_path: str, test_data_path: str):
        train_df = pd.read_csv(f"{train_data_path}/{dataset_name}.csv")
        y_train = train_df.pop(train_df.columns[-1])
        X_train = train_df

        test_df = pd.read_csv(f"{test_data_path}/{dataset_name}.csv")
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