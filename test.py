from pymfe.mfe import MFE
import numpy as np
import pandas as pd
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from sklearn.preprocessing import StandardScaler


def load_train_test_data(dataset_name: str, train_data_path: str, test_data_path: str):
    train_df = pd.read_csv(f"{train_data_path}/{dataset_name}.csv")
    y_train = train_df.pop(train_df.columns[-1])
    X_train = train_df

    test_df = pd.read_csv(f"{test_data_path}/{dataset_name}.csv")
    y_test = test_df.pop(test_df.columns[-1])
    X_test = test_df

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

def check_convert_data_type(data):
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    return data

def extract_meta_features(mfe, X, y=None):
    X = check_convert_data_type(X)

    if y is None:
        mfe.fit(X, suppress_warnings=True)
    else:
        y = check_convert_data_type(y)
        mfe.fit(X, y, suppress_warnings=True)

    columns_and_features = mfe.extract(cat_cols="auto", suppress_warnings=False, verbose=0)
    columns = columns_and_features[0]
    features = columns_and_features[1]
    
    features = np.nan_to_num(features).tolist()
    for i in range(0, len(features)):
        if features[i] > np.finfo(np.float32).max:
            features[i] = np.finfo(np.float32).max

    return columns, features

def test_mfe():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='mfe_test.log', encoding='utf-8', level=logging.DEBUG)

    complete_data_path = "./data/full_set/"
    dataset_list = [csv for csv in os.listdir(complete_data_path) if csv.endswith(".csv")]
    
    mfe = MFE(random_state=42)
    li = []
    for i, dataset in enumerate(dataset_list):
        dataset_name = dataset.split(".csv")[0]

        # Meta-Features extraction
        dt = pd.read_csv(complete_data_path + dataset)
        dt = dt.dropna()

        y = dt.pop('class')
        X = dt

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        columns, features = extract_meta_features(mfe, X, y)

        logger.info(f"Finished: {dataset}\tlen columns {len(columns)} | len features {len(features)}")

def test_clf():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='clf_test.log', encoding='utf-8', level=logging.DEBUG)

    complete_data_path = "./data/full_set/"
    dataset_list = [csv for csv in os.listdir(complete_data_path) if csv.endswith(".csv")]
    
    mfe = MFE(random_state=42)
    li = []
    for i, dataset in enumerate(dataset_list):
        dataset_name = dataset.split(".csv")[0]
                
        X_train, y_train, X_test, y_test = load_train_test_data(dataset_name, "./data/train_set/", "./data/test_set/")

        clf = None
        clf = LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000)
        
        calib_clf = CalibratedClassifierCV(clf, cv=3, n_jobs=-1)
        calib_clf.fit(X_train, y_train)

        scores = getTrainingScores(X_train, y_train, 10, clf)[0]
        tprfpr = getTPRFPR(scores)
        clf.fit(X_train, y_train)
           
        logger.info(f"Finished: {dataset}")

if __name__ == "__main__":
    test_clf()