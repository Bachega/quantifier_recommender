import os
import pandas as pd
import numpy as np
import json
import time
import joblib
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

class QuantifierEvaluator:
    __quantifiers = ['CC',
                     'ACC',
                     'PACC',
                     'PCC',
                     'SMM',
                     'HDy',
                     'DyS',
                     'SORD',
                     'MS',
                     'MAX',
                     'X']
    
    def __init__(self) -> None:
        self.evaluation_table = None
        self._processed_datasets = {}
        for quantifier in self.__quantifiers:
            self._processed_datasets[quantifier] = []
        

    def load_processed_datasets(self, path = "./data/processed_datasets.json"):
        if(os.path.isfile(path)):
            with open(path, "r") as file:
                self._processed_datasets = json.load(file)
    
    def save_processed_datasets(self,  path = "./data/processed_datasets.json"):
            with open(path, 'w') as file:
                json.dump(self._processed_datasets, file, indent=4, ensure_ascii=False)
    
    def __dataset_already_processed(self, quantifier, dataset_path):
        self.load_processed_datasets()
        return dataset_path in self._processed_datasets[quantifier]
    
    def __load_train_test_data(self, dataset_name):
        train_df = pd.read_csv(f"./data/train_data/{dataset_name}.csv")
        y_train = train_df.pop(train_df.columns[-1])
        X_train = train_df

        test_df = pd.read_csv(f"./data/test_data/{dataset_name}.csv")
        y_test = test_df.pop(test_df.columns[-1])
        X_test = test_df

        return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    def evaluate_quantifier(self, quantifier, dataset_name):
        X_train, y_train, X_test, y_test = self.__load_train_test_data(dataset_name)

        clf = None
        try:
            clf = joblib.load(f"./data/estimator_parameters/{dataset_name}.joblib")
            clf.n_jobs = -1
        except:
            clf = LogisticRegression(random_state=42, n_jobs=-1)
        
        calib_clf = CalibratedClassifierCV(clf, cv=3, n_jobs=-1)
        calib_clf.fit(X_train, y_train)

        scores = getTrainingScores(X_train, y_train, 10, clf)[0]
        tprfpr = getTPRFPR(scores)
        clf.fit(X_train, y_train)

        niterations = 10
        batch_sizes = list([100])
        alpha_values = [round(x, 2) for x in np.linspace(0,1,20)]

        pos_scores = scores[scores["class"]==1]["scores"]
        neg_scores = scores[scores["class"]==0]["scores"]

        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test, columns=[str(len(X_test.columns))])
        df_test = pd.concat([X_test, y_test], axis=1)

        df_test_pos = df_test.loc[df_test[df_test.columns[-1]] == 1]
        df_test_neg = df_test.loc[df_test[df_test.columns[-1]] == 0]

        for sample_size in batch_sizes:
            for alpha in alpha_values:
                error = []

                # Repeats the same experiment (to reduce variance)
                for iter in range(niterations):
                    pos_size = int(round(sample_size * alpha, 2))
                    neg_size = sample_size - pos_size
                    
                    sample_test_pos = df_test_pos.sample( int(pos_size), replace = False)
                    sample_test_neg = df_test_neg.sample( int(neg_size), replace = False)
                    
                    sample_test = pd.concat([sample_test_pos, sample_test_neg])
                    test_label = sample_test[sample_test.columns[-1]]

                    test_sample = sample_test.drop([sample_test.columns[-1]], axis=1) # sample_test.drop(["class"], axis=1)  #dropping class label columns
                    te_scores = clf.predict_proba(test_sample)[:,1]  #estimating test sample scores

                    n_pos_sample_test = list(test_label).count(1) #Counting num of actual positives in test sample
                    calcultd_pos_prop = round(n_pos_sample_test/len(sample_test), 2) #actual pos class prevalence in generated sample

                    #..............Test Sample QUAPY exp...........................
                    te_quapy = None
                    external_qnt = None

                    
                    #.............Calling of Methods..................................................
                    start = time.time()
                    pred_pos_prop = apply_quantifier(qntMethod=quantifier,
                                                    clf=calib_clf,
                                                    scores=scores,
                                                    p_score=pos_scores,
                                                    n_score=neg_scores,
                                                    train_labels=None,
                                                    test_score=te_scores,
                                                    TprFpr=tprfpr,
                                                    thr=0.5,
                                                    measure='hellinger',
                                                    test_data=test_sample,
                                                    test_quapy=te_quapy,
                                                    external_qnt=external_qnt) #y_test=test_label
                    stop = time.time()
                    t = stop - start

                    pred_pos_prop = np.round(pred_pos_prop,2)  #predicted class proportion
                    
                    #..............................RESULTS Evaluation.....................................
                    abs_error = round(abs(calcultd_pos_prop - pred_pos_prop), 2) # absolute error