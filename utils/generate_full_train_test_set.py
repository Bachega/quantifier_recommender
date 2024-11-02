import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# This function generates the train and test partitions
# using holdout with test_size = 0.3 AND random_state = 42
# Also: data can be normalized (or not) using minmax or zscore
def generate_full_train_test_set(source_path: str, full_dest_path: str, train_dest_path: str,
                                 test_dest_path: str, scaling_method: str = None,
                                 test_size=0.3, random_state=42):
    assert scaling_method is None or scaling_method in ["minmax", "zscore"], "scaling_method must be None, 'minmax' or 'zscore'"
    
    scaler = None
    if scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif scaling_method == "zscore":
        scaler = StandardScaler()

    dataset_list = [csv for csv in os.listdir(source_path) if csv.endswith(".csv")]

    if not os.path.exists(full_dest_path):
        os.makedirs(full_dest_path)

    if not os.path.exists(train_dest_path):
        os.makedirs(train_dest_path)
    
    if not os.path.exists(test_dest_path):
        os.makedirs(test_dest_path)
    
    for dataset_name in dataset_list:
        dataset = pd.read_csv(f"{source_path}/{dataset_name}")
        
        columns = dataset.columns
        y = dataset[columns[-1]].values
        X = dataset.drop(columns[-1], axis=1).values
        # y = dataset.pop(dataset.columns[-1])
        if scaler:
            X = scaler.fit_transform(X)
        # else:
        #     X = dataset.values
        X = np.c_[X, y]
        dataset_transformed = pd.DataFrame(data=X, columns=columns)

        train, test = train_test_split(dataset_transformed, test_size=test_size, random_state=random_state)

        dataset.to_csv(f"{full_dest_path}/{dataset_name}", index=False)
        train.to_csv(f"{train_dest_path}/{dataset_name}", index=False)
        test.to_csv(f"{test_dest_path}/{dataset_name}", index=False)