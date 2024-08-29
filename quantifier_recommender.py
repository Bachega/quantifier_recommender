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

from meta_feature_extractor import MetaFeatureExtractor
    
class QuantifierRecommender():
    def __init__(self, supervised: bool = True, meta_table = None):
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