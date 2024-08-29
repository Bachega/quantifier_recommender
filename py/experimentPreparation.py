import pandas as pd
import numpy as np
import argparse
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils.experimentList import Experiments as Exp
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.getCalibratedTrainingScores import getCalibratedTrainingScores
from quantifiers.PWKCLF import PWKCLF
from utils.fitQuantifierSchumacherGithub import fitQuantifierSchumacherGithub
import os
import quapy as qp

import pdb


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def experimentPreparation(dataset, trainedArtifacts_path='trainedArtifacts/'):
   """
   This function estimates the True positive rate(TPR), False positive rate(FPR), Scores, learned Classifier and also training and test partition from the provided dataset. 
    
   Parameters
   ---------
   dataset : string
       name of a dataset (user input)

    Returns
    ---------
    Numeric vector of Scores

    Matrix of TPR and FPR

    Learned Classifier

    Training Partition

    Test Partition
   """

   folds_num = 10               #number of CV-folds for evaluating scores
   niterations = 10
   #....................input / output path directories.......................
   exp = Exp[dataset]
   dataset_filename = exp["input"]
   class_feature = exp["class_feature"]
   positive_label = exp["positive_label"]

    #........................Reading data.................................
   print(dataset_filename)
   
   folder = trainedArtifacts_path + dataset   #folder path to save the results for each dataset
    
   data = pd.read_csv(dataset_filename, index_col=False)

   #.............. DATA PREPARATION................................................
   neg_label = list(x for x in set(data[class_feature]) if x!= positive_label) #extracting negative labels
   print('negatives _labels', neg_label) 
   all_labels = neg_label + [positive_label]
   print('All labels in dataset', all_labels)
   
   data_df = pd.DataFrame(data.loc[data[class_feature].map(lambda x: x in all_labels)])
   binary_label = "Binary_label"
   data_df[binary_label] = data_df.apply(lambda x: 1 if x[class_feature] == positive_label else 0, axis=1) # creating new binary label column
   
   
   for iter in range(niterations):
        os.makedirs( folder + '/%d' % (iter+1) , exist_ok=True)
        path = folder + '/%d' % (iter+1)

        #.......................Train_Test_Split........................................
   
        X_train, X_test, y_train, y_test = train_test_split( data_df, data_df[binary_label], test_size=0.5, shuffle = True, stratify = data_df[binary_label])

        df_train_pos = X_train.loc[X_train["Binary_label"] == 1] # seperating positive test examples
        df_train_neg = X_train.loc[X_train["Binary_label"] == 0] # seperating negative test examples
        
        #.............. New training Sample........................
        train_sample_size = min(len(df_train_pos), len(df_train_neg))
   
        train_prop = [0.05,0.1,0.3,0.5,0.7,0.9] 
        
        for alpha in train_prop:
            pos_size = np.int(round(train_sample_size * alpha, 2))  # number of positive observation
            neg_size = train_sample_size - pos_size

            sample_train_pos = df_train_pos.sample( int(pos_size), replace = False)
            sample_train_neg = df_train_neg.sample( int(neg_size), replace = False)

            sample_train = pd.concat([sample_train_pos, sample_train_neg])
            sample_train_label = sample_train[binary_label]

            print(len(df_train_pos), len(df_train_neg), train_sample_size)
            
            sample_tr = pd.DataFrame(sample_train).drop([class_feature, binary_label], axis=1) #training set (excluding label info columns)

            #.....Defining the scorer/model for predicting test scores............
            model = RandomForestClassifier(n_estimators=200)
            #.................................................................................

            #.................. Estimating training Scores and Fitting the scorer/classifier... 
            training_scores, clf = getTrainingScores(sample_tr, sample_train_label, folds_num, model)

            #.................. Estimating Calibrated Training scores ......................
            calibrated_training_scores, calibrate_clf = getCalibratedTrainingScores(sample_tr, sample_train_label, folds_num, model)

            #...................Estimating TPR & FPR........................................
            tprfpr = getTPRFPR(training_scores)

            #...................Estimating TPR & FPR for calibrated scores..................
            calb_tprfpr = getTPRFPR(calibrated_training_scores)

            #................. Saving the data .............................................
            os.makedirs( path + '/%f' % alpha , exist_ok=True)
            new_path = path + '/%f' % alpha 

             #..................Schumacher Paper Methods....................
            schumi_quantifiers = ['Readme', 'HDx', 'FormanMM', 'CDE']
            for method_name in schumi_quantifiers:
                schumacher_qnt = fitQuantifierSchumacherGithub(method_name, sample_tr, sample_train_label)
                with open(new_path+ '/'+method_name+'_model_%s'%dataset +'_%f'%alpha + '.pkl', mode='wb') as out:        #saving the learned scorer/model
                    pickle.dump(schumacher_qnt, file= out)
            
            #..................PWK from Schumacher library....................
            #model_pwk=PWKCLF(alpha=1, n_neighbors=10, algorithm="auto", metric="euclidean", leaf_size=30, p=2, metric_params=None, n_jobs=None)
            #clf_pwk = model_pwk.fit(sample_tr, sample_train_label)
            #with open(new_path+ '/PWK_model_%s'%dataset +'_%f'%alpha + '.pkl', mode='wb') as out:        #saving the learned scorer/model
            #    pickle.dump(clf_pwk, file= out)
            #..................EMQ from quapy library....................
            tr_quapy = qp.data.LabelledCollection(sample_tr, sample_train_label)            
            mod_quapy = qp.method.aggregative.EMQ(model)
            mod_quapy.fit(tr_quapy)
            with open(new_path+ '/EMQ_model_%s'%dataset +'_%f'%alpha + '.pkl', mode='wb') as out:        #saving the learned scorer/model
                pickle.dump(mod_quapy, file= out)
            
            pd.DataFrame(sample_train.to_csv(new_path + '/train_data_%s'%dataset + '_%f'%alpha + '.csv', index = False))   #saving sample from the partition of training data
            pd.DataFrame(X_test.to_csv(new_path + '/test_data_%s'%dataset + '_%f'%alpha + '.csv', index = False))    #saving partition of test data
            tprfpr.to_csv(new_path + '/tprfpr_%s'%dataset + '_%f'%alpha + '.csv', index=False)         #saving tpr and fpr 
            calb_tprfpr.to_csv(new_path + '/calibrated_tprfpr_%s'%dataset + '_%f'%alpha + '.csv', index=False)         #saving tpr and fpr 
            training_scores.to_csv(new_path +'/scores_training_%s'%dataset + '_%f'%alpha +'.csv', index = False)  #Saving training scores
            pd.DataFrame(calibrated_training_scores).to_csv(new_path +'/calibrated_scores_training_%s'%dataset + '_%f'%alpha +'.csv', index = False)    #saving calibrated training scores

            with open(new_path +'/model_%s'%dataset +'_%f'%alpha + '.pkl', mode='wb') as out:        #saving the learned scorer/model
               pickle.dump(clf, file= out)
            
            with open(new_path +'/calibrated_model_%s'%dataset +'_%f'%alpha + '.pkl', mode='wb') as out:        #saving the calibrated scorer/model
                pickle.dump(calibrate_clf, file= out)
        

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('exp', type=str)
  args = parser.parse_args()
  experimentPreparation(args.exp)