from pymfe.mfe import MFE
import numpy as np
import pandas as pd

class MetaFeatureExtractor:
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

        columns_and_features = self.mfe.extract(cat_cols="auto", suppress_warnings=True, verbose=0)
        columns = columns_and_features[0]
        features = columns_and_features[1]
        
        features = np.nan_to_num(features).tolist()
        for i in range(0, len(features)):
            if features[i] > np.finfo(np.float32).max:
                features[i] = np.finfo(np.float32).max

        return columns, features