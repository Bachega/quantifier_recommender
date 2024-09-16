import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pdb

# This function generates the train and test partitions
# using holdout with test_size = 0.3 AND random_state = 42
# Also: data is normalized (MinMax Scale)
def generate_train_test_data(source_path,
                             train_dest_path = "./data/train_data",
                             test_dest_path = "./data/test_data",
                             test_size=0.3,
                             random_state=42):
    
    dataset_list = [csv for csv in os.listdir(source_path) if csv.endswith(".csv")]
    scaler = MinMaxScaler()

    if not os.path.exists(train_dest_path):
        os.makedirs(train_dest_path)
    
    if not os.path.exists(test_dest_path):
        os.makedirs(test_dest_path)
    
    for dataset_name in dataset_list:
        dataset = pd.read_csv(f"{source_path}/{dataset_name}")
        
        dataset = dataset.dropna()

        columns = dataset.columns
        y = dataset.pop(dataset.columns[-1])
        X = scaler.fit_transform(dataset)
        X = np.c_[X, y]
        dataset = pd.DataFrame(data=X, columns=columns)

        train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)

        train.to_csv(f"{train_dest_path}/{dataset_name}", index=False)
        test.to_csv(f"{test_dest_path}/{dataset_name}", index=False)

# Hyperparameters of LogisticRegression (LR) are tuned using the train set
# LR is used as a scorer for the quantifiers
def grid_search(X_train, y_train):
    clf = LogisticRegression(penalty="l2", random_state=42, max_iter=10000)

    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    class_weight = [None, "balanced"]
    
    grid = {"C": C,
            "class_weight": class_weight}
    
    search = GridSearchCV(estimator=clf,
                          param_grid=grid,
                          cv=3,
                          verbose=2,
                          n_jobs=-1)
    
    search.fit(X_train, y_train)
    return search.best_estimator_

# This function loads the Train and Test data created in the
# generate_train_test_data function
def load_train_test_data(dataset_name, train_data_path, test_data_path):
    train_df = pd.read_csv(f"{train_data_path}/{dataset_name}.csv")
    y_train = train_df.pop(train_df.columns[-1])
    X_train = train_df

    test_df = pd.read_csv(f"{test_data_path}/{dataset_name}.csv")
    y_test = test_df.pop(test_df.columns[-1])
    X_test = test_df

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()