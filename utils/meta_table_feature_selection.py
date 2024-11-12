import pandas as pd
import numpy as np
import sklearn.feature_selection as fs

# CHI-SUQRE --> FOR CATEGORICAL DATA


def variance_treshold(X):
    selector = fs.VarianceThreshold()
    _ = selector.fit_transform(X)
    
    # Get the support mask and apply it to the dataframe columns
    support = selector.get_support(indices=True)
    reduced_df = X.iloc[:, support]
    
    print(f"NOT PREPROCESSED: {X.shape[1]}")
    print(f"VARIANCE THRESHOLD: {reduced_df.shape[1]}")
    
    return reduced_df

def cfs(df_X, df_y, min_corr = 0.05):
    df_X['class'] = df_y.values
    corr_matrix = df_X.corr()
    corr_with_class = corr_matrix['class']
    to_remove = corr_with_class[(corr_with_class > -min_corr) & (corr_with_class < min_corr)].index
    df_X = df_X.drop(columns=to_remove)
    df_X = df_X.drop(columns=['class'])
    return df_X

def info_gain(X, y):
    pass

def rfe(X, y):
    pass

if __name__ == "__main__":
    df_X = pd.read_csv("./meta_features_table.csv", index_col=0)
    df_y = pd.read_csv("./evaluation_table.csv", index_col=[0, 1])
    quantifiers = list(set(x[0] for x in df_y.index))

    # VARIANCE THRESHOLD
    df_X = variance_treshold(df_X)

    selected_columns = []
    for qtf in quantifiers:
        df_X_reduced = cfs(df_X, df_y.loc[qtf]['abs_error'])
        selected_columns.append(df_X_reduced.columns.tolist())
    
    import pdb; pdb.set_trace()