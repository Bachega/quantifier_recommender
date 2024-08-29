from timeit import default_timer as timer
import pandas as pd
import numpy as np
import pickle

import argparse
from utils.methodlist import methods
import sys
from utils.applyquantifiers import apply_quantifier
from utils.fCalculate_MRAE import fCalculateRAE

import quapy as qp
import os

from quantifiers.PWKCLF import PWKCLF


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def experimentRunner(dataset, method_name, niterations = 10, calibrated = False, trainedArtifacts_path="./trainedArtifacts/"):
   """
   This is the main function to run the experimental setup. It returns performance of quantification methods in terms MAE across different test sample sizes and training proportions and save the results in .CSV file.
    
   Parameters
   ---------
   dataset : string
       name of a dataset (user input)
   method_name : string
       name of the method (user input)
   niterations : integer
        number of iterations to repeat the experiment

    Returns
    ---------
    CSV file
        that contains performance of Quantification method in terms of MAE.
    
    NOTE : To run experiemnt for any method use the following in the command prompt
            python Experiment_new_setup.py spambase smm --it=10
   """

    #>>>>>>>..............Experimental_setup............>>>>>>>>>>
   vdist = ["topsoe", "jensen_difference", "prob_symm", "ord", "sord", "hellinger"] 
   names_vdist = ["TS", "JD", "PS", "ORD", "SORD", "HD"] 
   counters    = ["HDy","DyS-TS","SORD", "MS", "CC", "ACC","SMM"]
   measure     = "topsoe"                   #default measure for DyS
   external_quantifiers = ['Readme', 'HDx', 'FormanMM', 'CDE', 'EMQ', 'PWK']
   print('OK')
   method_kargs = methods[method_name]["kargs"]
   external_qnt = None
   te_quapy = None
   pos_scores = None
   neg_scores = None
   clf = None
   scores = pd.DataFrame(columns=['scores', 'class'])
   tprfpr = None

   result_path = "results/raw/"                 #Foldername for saving the results
   os.makedirs( result_path , exist_ok=True) 
   print('Calibrated -- > '+ str(calibrated))
   table2=pd.DataFrame() 
   
   tr_prop = [0.05,0.1,0.3,0.5,0.7,0.9]         # Training proportions used to train the classifier
   tr_sample= 10                               # Number of Iterations for training proportion
   for train_prop in tr_prop:
       for train_sample in range(tr_sample):
            folder = trainedArtifacts_path + dataset +'/%d' % (train_sample+1) + '/%f' % train_prop 
            tr = pd.read_csv(folder + '/train_data_%s' % dataset + '_%f'%train_prop + '.csv', index_col=False, engine='python')
            te = pd.read_csv(folder + '/test_data_%s' % dataset + '_%f'%train_prop + '.csv', index_col=False, engine='python')
            if method_name in external_quantifiers:
                if method_name == 'PWK':
                    clf_pwk=PWKCLF(alpha=1, n_neighbors=10, algorithm="auto",
                               metric="euclidean", leaf_size=30, p=2,
                               metric_params=None, n_jobs=None)

                    external_qnt = clf_pwk.fit(tr.drop(["class","Binary_label"], axis=1), tr['Binary_label'])
                else:
                    external_qnt = pickle.load(open(folder +'/'+method_name+'_model_%s' % dataset +'_%f'%train_prop + '.pkl', 'rb')) 
            else:
                if calibrated == 'True':
                    tprfpr = pd.read_csv(folder + '/calibrated_tprfpr_%s'% dataset + '_%f'%train_prop + '.csv', index_col = False, engine='python')
                    scores = pd.read_csv(folder +'/calibrated_scores_training_%s' % dataset + '_%f'%train_prop +'.csv', index_col=False, engine='python')
                    clf = pickle.load(open(folder +'/calibrated_model_%s' % dataset +'_%f'%train_prop + '.pkl', 'rb')) 
                else:
                    tprfpr = pd.read_csv(folder + '/tprfpr_%s'% dataset + '_%f'%train_prop + '.csv', index_col = False, engine='python')                
                    scores = pd.read_csv(folder +'/scores_training_%s' % dataset + '_%f'%train_prop +'.csv', index_col=False, engine='python')
                    clf = pickle.load(open(folder +'/model_%s' % dataset +'_%f'%train_prop + '.pkl', 'rb'))
                
                pos_scores = scores[scores["class"]==1]["scores"]   #separating positve scores from training scores  
                neg_scores = scores[scores["class"]==0]["scores"]           
            
            df_test = pd.DataFrame(te)
            print('test_lenght',len(df_test))
            
            df_test_pos = df_test.loc[df_test["Binary_label"] == 1] # seperating positive test examples
            df_test_neg = df_test.loc[df_test["Binary_label"] == 0] # seperating negative test examples
            
            max_allowed = min(len(df_test_pos), len(df_test_neg))
            batch_sizes = list(range(10, min(91, max_allowed + 1), 10)) + list(range(100, min(501, max_allowed + 1), 100))            
            alpha_values = [0, 0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]   #Test class proportion
        
            table=pd.DataFrame()
            for sample_size in batch_sizes:   #[10,100,500], batch_sizes, Varying test set sizes                
                for alpha in alpha_values: #   Varying positive class distribution
                    abs_error = []
                    error = []
                    pred_pos_prop=[]
                    ms_per_example = []
                    for iter in range(niterations):
                        print('Sample size #%d' % (sample_size))
                        print('iteration #%d' % (iter + 1))

                        pos_size = np.int(round(sample_size * alpha, 2))
                        neg_size = sample_size - pos_size

                        sample_test_pos = df_test_pos.sample( int(pos_size), replace = False)
                        sample_test_neg = df_test_neg.sample( int(neg_size), replace = False)
                        
                        sample_test = pd.concat([sample_test_pos, sample_test_neg])
                        #print ('Te(+) :', len(sample_test_pos), 'Te(-) :', len(sample_test_neg),'Te :', len(sample_test))
                        
                        test_label = sample_test["Binary_label"]                        
                        test_sample = sample_test.drop(["class","Binary_label"], axis=1)  #dropping class label columns

                        te_scores = None
                        if method_name not in external_quantifiers:
                            te_scores = clf.predict_proba(test_sample)[:,1]  #estimating test sample scores

                        #..............Test Sample QUAPY exp...........................
                        if method_name == 'EMQ':
                            te_quapy = qp.data.LabelledCollection(sample_test.drop(["class","Binary_label"], axis=1), test_label)                        

                        n_pos_sample_test = list(test_label).count(1) #Counting num of actual positives in test sample
                        calcultd_pos_prop = round(n_pos_sample_test/len(sample_test), 2) #actual pos class prevalence in generated sample

                        #.............Calling of Methods.................................................. 
                        pred_pos_prop = apply_quantifier(qntMethod= method_name, 
                                                        clf = clf,
                                                        scores = scores['scores'], 
                                                        p_score=pos_scores,
                                                        n_score=neg_scores,
                                                        train_labels = scores['class'], 
                                                        test_score = te_scores, 
                                                        TprFpr = tprfpr, 
                                                        thr = 0.5, 
                                                        measure = measure,
                                                        test_data = test_sample, 
                                                        test_quapy = te_quapy, 
                                                        external_qnt = external_qnt)                        

                        pred_pos_prop = np.round(pred_pos_prop,2)  #predicted class proportion                        
                        #..............................RESULTS Evaluation.....................................                        
                        abs_error = np.round(abs(calcultd_pos_prop - pred_pos_prop),2) #absolute error
                        error = np.round(calcultd_pos_prop - pred_pos_prop , 2)     # simple error Biasness
                        ms_per_example = -1 #(tm_end - tm_start) * 1000 / len(sample_test) #time calculation
                        
                        table = table.append(pd.DataFrame([(train_sample+1),train_prop,(iter + 1),sample_size,alpha,calcultd_pos_prop,pred_pos_prop,abs_error,error,fCalculateRAE(abs_error,calcultd_pos_prop),ms_per_example,method_name,dataset]).T)

            table.columns = ["Train_sample","Train_prop","Test_sample#","Test_size","alpha","actual_prop","pred_prop","abs_error","error/bias","relative_error","time","quantifier","dataset"]
            table2 = table2.append(table)
            
   table2.to_csv(result_path +'/' + dataset + '_%s' % method_name + '.csv', index = False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('exp', type=str)
  parser.add_argument('method', type=str)
  parser.add_argument('--it', type=int, default=10)
  parser.add_argument('cal', type=str)
  
  args = parser.parse_args()
  experimentRunner(dataset=args.exp, method_name=args.method, niterations=args.it, calibrated=args.cal)