from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os
import joblib

from ..extractor import MetaFeatureExtractor
from ..evaluator import QuantifierEvaluator

class BaseRecommender(ABC):
    def __init__(self, supervised = True, load_default = True):
        self._load_default = load_default
        self.mfe = MetaFeatureExtractor()
        self.quantifier_evaluator = QuantifierEvaluator()
        self.supervised = supervised
    
    @abstractmethod
    def _fit_method(self, meta_features_table, not_aggregated_evaluation_table, evaluation_table):
        pass

    def fit(self, full_set_path: str, train_set_path: str, test_set_path: str) -> None:
        dataset_list = [csv for csv in os.listdir(full_set_path) if csv.endswith(".csv")]
        class_name = self.__class__.__name__

        if class_name == "KNNRecommender":
            func_type = "utility"
        elif class_name == "RegressionRecommender":
            func_type = "cost"

        meta_features_table = None
        evaluation_list = []
        for i, dataset in enumerate(dataset_list):
            dataset_name = dataset.split(".csv")[0]
            dt = pd.read_csv(full_set_path + dataset)

            ################################################## Meta-Features extraction ##################################################
            if self.supervised:
                y = dt.pop('class')
            else:
                y = None
            X = dt
            
            columns, features = self.mfe.extract_meta_features(X, y)
            if meta_features_table is None:
                meta_features_table = pd.DataFrame(columns=columns)
            meta_features_table.loc[dataset_name] = features

            ################################################## Quantifiers evaluation ##################################################
            X_train, y_train, X_test, y_test = self._load_train_test_set(dataset_name, train_set_path, test_set_path)
            evaluation_list.append(self.quantifier_evaluator.evaluate_quantifiers(dataset_name=dataset_name,
                                                                                    X_train=X_train,
                                                                                    y_train=y_train,
                                                                                    X_test=X_test,
                                                                                    y_test=y_test,
                                                                                    func_type=func_type))
            # # DELETE THIS
            # if i == 3:
            #     break
        
        not_aggregated_evaluation_table = pd.concat(evaluation_list, axis=0)
        evaluation_table = self._aggregate_evaluation_table(not_aggregated_evaluation_table, func_type)
        
        self._fit_method(meta_features_table, not_aggregated_evaluation_table, evaluation_table)
    
    def _load_train_test_set(self, dataset_name: str, train_set_path: str, test_set_path: str):
        train_df = pd.read_csv(f"{train_set_path}/{dataset_name}.csv")
        y_train = train_df.pop(train_df.columns[-1])
        X_train = train_df

        test_df = pd.read_csv(f"{test_set_path}/{dataset_name}.csv")
        y_test = test_df.pop(test_df.columns[-1])
        X_test = test_df

        return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    
    def _aggregate_evaluation_table(self, not_aggregated_evaluation_table, func_type):
        evaluation_table = not_aggregated_evaluation_table.sort_values(by=['quantifier', 'dataset'])

        if func_type == "utility":
            evaluation_table = not_aggregated_evaluation_table.sort_values(by=['quantifier', 'dataset'])
            evaluation_table = evaluation_table.groupby(["quantifier", "dataset"]).agg(
                inv_abs_error = pd.NamedAgg(column="inv_abs_error", aggfunc="mean"),
                run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
            )
        elif func_type == "cost":
            evaluation_table = evaluation_table.groupby(["quantifier", "dataset"]).agg(
                abs_error = pd.NamedAgg(column="abs_error", aggfunc="mean"),
                run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
            )
        
        return evaluation_table

    def persist_model(self, path: str = None):
        if not path:
            path = f"{self.__class__.__name__}.joblib"
        with open(path, "wb") as file_handler:
            joblib.dump(self, file_handler)

    def load_model(self, path: str = None):
        if not path:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", f"{self.__class__.__name__}.joblib")
        model = joblib.load(path)
        self.__dict__.update(model.__dict__)