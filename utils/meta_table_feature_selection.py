import pandas as pd
import numpy as np
import sklearn.feature_selection as fs

def meta_table_feature_selection():
    meta_features_table = pd.read_csv("./meta_features_table.csv", index_col=0)
    X = meta_features_table.values
    print(f"NOT PREPROCESSED: {len(X[0])}")
    
    selector = fs.VarianceThreshold()
    X = selector.fit_transform(X)

    print(f"VARIANCE THRESHOLD: {len(X[0])}")


    # CFS, CHI-SQUARE, INFO-GAIN, RFE.

if __name__ == "__main__":
    meta_table_feature_selection()