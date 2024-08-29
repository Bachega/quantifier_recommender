import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import random
import pdb
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler
import pickle

class MetaFeatureExtractor():
    def __init__(self, random_state: int = 42):
        self.mfe = MFE(random_state=random_state)
    
    def __check_convert_data_type(self, data):
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        return data

    def extract_meta_features(self, X, y=None):
        X = self.__check_convert_data_type(X)

        if y is None:
            self.mfe.fit(X)
        else:
            y = self.__check_convert_data_type(y)
            self.mfe.fit(X, y)

        columns_and_features = self.mfe.extract(cat_cols='auto', suppress_warnings=True, verbose=0)
        columns = columns_and_features[0]
        features = columns_and_features[1]
        
        features = np.nan_to_num(features).tolist()
        for i in range(0, len(features)):
            if features[i] > np.finfo(np.float32).max:
                features[i] = np.finfo(np.float32).max

        return columns, features
    
class QuantifierRecommender():
    def __init__(self, supervised:bool = True, meta_table = None):
        self.meta_table = meta_table
        self.mfe = MetaFeatureExtractor()
    
    def __initialize_meta_table(self, columns: list):
        self.meta_table = pd.DataFrame(columns=columns)

    def load_meta_table(self, meta_table_path) -> None:
        self.meta_table.read_csv(meta_table_path)
    
    def save_meta_table(self, meta_table_path: str = './meta-table.csv'):
        self.meta_table.to_csv(meta_table_path, index=False)

    def __normalize_meta_table(self) -> None:
        columns = self.meta_table.columns
        data = self.meta_table.values
        scaler = MinMaxScaler()
        scaler.fit(data)
        self.meta_table = pd.DataFrame(scaler.transform(data), columns=columns)
    
    def __extract_and_append(self, X, y=None):
        columns, features = self.mfe.extract_meta_features(X, y)

        if self.meta_table is None:
            self.__initialize_meta_table(columns)

        self.meta_table.loc[len(self.meta_table.index)] = features

    def construct_meta_table(self, dataset_path: str, supervised: bool = False):
        files = [csv for csv in os.listdir(dataset_path) if csv.endswith(".csv")]

        for f in files:
            dt = pd.read_csv(dataset_path + f)
            dt = dt.dropna()
            
            if supervised:
                y = dt.pop(dt.columns[-1])
            else:
                y = None
            X = dt
            
            self.__extract_and_append(X=X, y=y)
            
        self.__normalize_meta_table()

def load_datasets(path):
    dt_list = []
    files = [csv for csv in os.listdir(path) if csv.endswith(".csv")]
    for f in files:
        dt = pd.read_csv(path + f)
        dt = dt.dropna()
        dt_list.append(dt)

    return dt_list

if __name__ == "__main__":
    dataset_path = './datasets/'
    recommender = QuantifierRecommender(supervised=True)

    # dt_list = load_datasets(dataset_path)
    # for dt in dt_list:
    #     y = dt.pop(dt.columns[-1])
    #     X = dt
    #     recommender.extract_and_append(X, y)
    # recommender.normalize_meta_table()

    recommender.construct_meta_table(dataset_path=dataset_path, supervised=True)
    recommender.save_meta_table('meta-features.csv')