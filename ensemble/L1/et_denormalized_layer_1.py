# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 12:10:34 2016

@author: Bishwarup
"""

import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from ml_metrics import log_loss

# set directories and random seed
parent_dir = "F:/Kaggle/BNP_Paribas/pipeLine4F/"
data_dir = os.path.join(parent_dir, "Data/")
eval_dir = os.path.join(parent_dir, "evals/")
test_dir = os.path.join(parent_dir, "tests/")
np.random.seed(201604)





if __name__ == '__main__':

    start_time = datetime.now()

    train_file = os.path.join(data_dir, "train_processed_v19.csv")
    test_file = os.path.join(data_dir, "test_processed_v19.csv")
    fold_file = os.path.join(parent_dir, "fold4f.csv")
    print "reading data files..."
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    fold_ids = pd.read_csv(fold_file)
    pred_name = "et_v19_1"
    eval_prediction_file = os.path.join(eval_dir, pred_name+"_eval.csv")
    test_prediction_file = os.path.join(test_dir, pred_name+"_test.csv")    
    
    train.fillna(-1, inplace = True)
    test.fillna(-1, inplace = True)
    
    feature_names = [f for f in train.columns if f not in ["ID", "target", "train_flag"]]    
    
    eval_matrix = pd.DataFrame(columns  = ["ID", pred_name])
    test_matrix = pd.DataFrame({"ID" : test["ID"]})
    
    for i in xrange(fold_ids.shape[1]):
        
        val_ids = fold_ids.ix[:, i].dropna()
        idx = train["ID"].isin(list(val_ids))
        trainingSet = train[~idx]
        validationSet = train[idx]

        X_train = trainingSet[feature_names].copy()
        X_valid = validationSet[feature_names].copy()
        y_train = np.array(trainingSet["target"].copy())
        y_valid = np.array(validationSet["target"].copy())        
        et = ExtraTreesClassifier(n_estimators = 2000,
                                    criterion = "entropy",
                                    max_depth = 50,
                                    max_features = 0.9,
                                    min_samples_split = 3,
                                    min_samples_leaf = 5,
                                    bootstrap = False,
                                    oob_score = False,
                                    random_state = 112,
                                    verbose = 0,
                                    n_jobs = -1)
                                
        et.fit(X_train, y_train)
                         
        preds = et.predict_proba(X_valid)[:, 1]
        ll = log_loss(validationSet["target"], preds)
        df = pd.DataFrame({"ID" : validationSet["ID"], pred_name : preds})
        eval_matrix = eval_matrix.append(df, ignore_index = True)
        print "fold : {} | logloss: {}".format(i+1, ll)        
        del trainingSet, validationSet, et, preds, ll, X_train, X_valid, y_train, y_valid
        gc.collect()
    
    X_train = train[feature_names].copy()
    y_train = np.array(train["target"].copy())
    et = ExtraTreesClassifier(n_estimators = 2000,
                                    criterion = "entropy",
                                    max_depth = 50,
                                    max_features = 0.9,
                                    min_samples_split = 3,
                                    min_samples_leaf = 5,
                                    bootstrap = False,
                                    oob_score = False,
                                    random_state = 112,
                                    verbose = 0,
                                    n_jobs = -1)
    
    et.fit(X_train, y_train)
    
    tpreds = et.predict_proba(test[feature_names])[:, 1]
    test_matrix[pred_name] = tpreds
    
    eval_matrix.to_csv(eval_prediction_file, index = False)
    test_matrix.to_csv(test_prediction_file, index = False)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print "elapsed time {}".format(duration)