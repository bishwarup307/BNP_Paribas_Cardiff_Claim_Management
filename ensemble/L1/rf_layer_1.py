# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:49:14 2016

@author: Bishwarup
"""

from __future__ import division
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml_metrics import log_loss


# global variables
parent_dir = "F:/Kaggle/BNP_Paribas/"
child_dir = parent_dir + "pipeLine4F/"
fold_index_file = child_dir + "fold4f.csv"
data_path = child_dir + "Partitions/"
train_prediction_path = child_dir + "evals/"
test_prediction_path  = child_dir + "tests/"
np.random.seed(201604)

if __name__ == '__main__':
    
    start_time = datetime.now()
    fold_ids = pd.read_csv(fold_index_file)
    
    # loop over 4 feature subsets
    for i in xrange(1):
        
        data_version = i + 1
        train_file = data_path + "trainP" + `i+1` + ".csv"
        test_file = data_path + "testP" + `i+1` + ".csv"
        model_version = "RF_P" + `i+1`
        print "Reading version {0} data files ...".format(i+1)
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        
        feature_names = [f for f in train.columns if f not in ["ID", "target"]]
        
        train.fillna(-1, inplace = True)
        test.fillna(-1, inplace = True)
        
        # container for meta features
        eval_matrix = pd.DataFrame(columns = ["Fold", "ID", "ground_truth", model_version])
        test_matrix = pd.DataFrame({"ID" : test["ID"]})
        
        # generate train meta features with
        # 4-fold cross_validation
        for j in xrange(fold_ids.shape[1]):
        
            fold = j + 1
            val_ids = fold_ids.ix[:, j].dropna()
            idx = train["ID"].isin(list(val_ids))
            
            trainingSet = train[~idx]
            validationSet = train[idx]
            
            rf = RandomForestClassifier(n_estimators = 2000,
                                        criterion = "entropy",
                                        max_depth = 50,
                                        max_features = 0.8,
                                        min_samples_split = 3,
                                        bootstrap = False,
                                        oob_score = False,
                                        random_state = 112,
                                        verbose = 0,
                                        n_jobs = -1)
            
            rf.fit(trainingSet[feature_names], np.array(trainingSet["target"]))                          
            preds = rf.predict_proba(validationSet[feature_names])[:, 1]
            ll = log_loss(np.array(validationSet["target"]), preds)
            print "# Data_version : {0} | Fold : {1} | log_loss : {2}".format(i+1, j+1, ll)
            df = pd.DataFrame({"Fold" : np.repeat((j + 1), validationSet.shape[0]) ,"ID" : validationSet["ID"], "ground_truth" : validationSet["target"], 
                                    model_version : preds})
            tmp_name = "P" + `data_version` + "_Fold_" + `fold` + "_valid.csv"
            tmp_file = train_prediction_path + "tmp/" + tmp_name
            df.to_csv(tmp_file, index  = False)
            eval_matrix = eval_matrix.append(df, ignore_index = True)
            del rf, trainingSet, validationSet, ll, df
            
        # generate test meta features
        # train on all training instances
        rf = RandomForestClassifier(n_estimators = 2000,
                                        criterion = "entropy",
                                        max_depth = 50,
                                        max_features = 0.8,
                                        min_samples_split = 3,
                                        bootstrap = False,
                                        oob_score = False,
                                        random_state = 112,
                                        verbose = 0,
                                        n_jobs = -1)
        
        rf.fit(train[feature_names], train["target"])
        tpreds = rf.predict_proba(test[feature_names])[:, 1]
        test_matrix[model_version] = tpreds
        
        # save meta features to disk
        train_out = train_prediction_path + model_version + "_eval.csv"
        test_out = test_prediction_path + model_version + "_test.csv"
        print "\nwriting train predictions to {0} ...".format(train_out)
        eval_matrix.to_csv(train_out, index = False)
        print "writing test predictions to {0} ...".format(test_out)
        test_matrix.to_csv(test_out, index = False)
        
        del eval_matrix, test_matrix, train, test, feature_names, train_file, test_file
    
    end_time = datetime.now()
    print('Duration: {0}'.format(end_time - start_time))


