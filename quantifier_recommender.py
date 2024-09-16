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
from quantifier_evaluator import QuantifierEvaluator
from utils_ import load_train_test_data
    
class QuantifierRecommender:
    def __init__(self, supervised: bool = True, meta_table = None):
        self.meta_table = meta_table
        self.meta_features_table = None
        self.evaluation_table = None
        self.mfe = MetaFeatureExtractor()
        self.quantifier_evaluator = QuantifierEvaluator()
     
    def __get_normalized_meta_features_table(self):
        columns = self.meta_features_table.columns
        data = self.meta_features_table.values
        scaler = MinMaxScaler()
        scaler.fit(data)
        return pd.DataFrame(scaler.transform(data), columns=columns)
    
    def __extract_and_append(self, X, y = None):
        columns, features = self.mfe.extract_meta_features(X, y)

        if self.meta_features_table is None:
            self.meta_features_table = pd.DataFrame(columns=columns)

        self.meta_features_table.loc[len(self.meta_features_table.index)] = features
    
    def __evaluate_and_append(self, dataset_name, X_train, y_train, X_test, y_test):
        new_evaluation = self.quantifier_evaluator.evaluate_internal_quantifiers(dataset_name,
                                                                                 X_train,
                                                                                 y_train,
                                                                                 X_test,
                                                                                 y_test)
        
        if self.evaluation_table is None:
            self.evaluation_table = new_evaluation.copy(deep=True)
        else:
            self.evaluation_table = pd.concat([self.evaluation_table, new_evaluation], ignore_index=True)

    def load_meta_table(self, meta_table_path):
        self.meta_table.read_csv(meta_table_path)
    
    def save_meta_table(self, meta_table_path: str = "./data/meta-table.csv"):
        self.meta_table.to_csv(meta_table_path, index=False)

    def construct_meta_table(self, datasets_path: str, train_data_path: str,
                             test_data_path: str,  supervised: bool = False):
        dataset_list = [csv for csv in os.listdir(datasets_path) if csv.endswith(".csv")]

        for dataset in dataset_list:
            # Meta-Features Extraction
            dt = pd.read_csv(datasets_path + dataset)
            dt = dt.dropna()
            
            if supervised:
                y = dt.pop(dt.columns[-1])
            else:
                y = None
            X = dt
            
            self.__extract_and_append(X=X, y=y)

            # Quantifiers evaluation
            dataset_name = dataset.split(".csv")[0]
            X_train, y_train, X_test, y_test = load_train_test_data(dataset_name, train_data_path, test_data_path)
            self.__evaluate_and_append(dataset_name, X_train, y_train, X_test, y_test)

        # Normalize the extracted meta-features and insert them in the Meta-table
        self.meta_table = self.__get_normalized_meta_features_table()

        # Sort and aggregate the quantifiers evaluations
        self.evaluation_table.sort_values(by=['quantifier', 'dataset'], inplace=True)

