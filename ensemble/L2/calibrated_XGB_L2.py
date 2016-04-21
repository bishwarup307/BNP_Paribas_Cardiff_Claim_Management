# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 02:09:04 2016

@author: Bishwarup
"""
import os
import gc
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from datetime import  datetime
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

parent_dir = "F:/Kaggle/BNP_Paribas/pipeLine4F/"
data_dir = os.path.join(parent_dir, "Data/")
eval_dir = os.path.join(parent_dir, "evals/")
test_dir = os.path.join(parent_dir, "tests/")
output_dir = os.path.join(parent_dir, "Submissions/")
np.random.seed(11)

def merge_all(id_, path, key = "ID"):
    merged_data = pd.DataFrame({key : id_})
    file_list = os.listdir(path)
    for files_ in file_list:
        candidate = pd.read_csv(os.path.join(path, files_))
        if "Fold" in candidate.columns:
            candidate.drop("Fold", axis = 1, inplace = True)
        if "ground_truth" in candidate.columns:
            candidate.drop("ground_truth", axis = 1, inplace = True)
        if "target" in candidate.columns:
            candidate.drop("target", axis = 1, inplace = True)
        assert (len(id_) == candidate.shape[0]), "{0} have differnt number of rows!".format(files_)
        merged_data = pd.merge(merged_data, candidate, on = key, how = "left")
        
    print "merged {0} files ..".format(len(file_list))        
    return merged_data
    
if __name__ == "__main__":
    print "reading L0 ..."
    train_file = os.path.join(data_dir, "train_processed_v19.csv")
    test_file = os.path.join(data_dir, "test_processed_v19.csv")
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    print "merging L1..."
    train_preds = merge_all(id_=np.array(train["ID"]),path=eval_dir)
    test_preds = merge_all(id_=np.array(test["ID"]),path=test_dir)
    
    print "merging L0 and L1..."
    train = pd.merge(train, train_preds, on = "ID", how = "left")
    test = pd.merge(test, test_preds, on = "ID", how = "left")
    
    feature_names = [f for f in train.columns if f not in ["ID", "target", "train_flag"]]
    
    skf = StratifiedKFold(np.array(train["target"]), n_folds = 5, shuffle = True, random_state = 14) 

    cv = []
    biter = []    
    for fold, (itr, icv) in enumerate(skf):
    
        print "------ Fold %d -----------\n" %(fold+1)
        
        trainingSet = train.iloc[itr]
        validationSet = train.iloc[icv]
        
        gbm = XGBClassifier(max_depth=8,
                            learning_rate = 0.01,
                            n_estimators=3000,
                            subsample=0.6,
                            colsample_bytree=0.3,
                            objective="binary:logistic",
                            silent = False,
                            min_child_weight=1,                       
                            nthread=-1)
                            
        gbm.fit(trainingSet[feature_names], np.array(trainingSet["target"]),
                eval_metric="logloss",
                eval_set=[(trainingSet[feature_names], np.array(trainingSet["target"])), (validationSet[feature_names], np.array(validationSet["target"]))],
                         early_stopping_rounds=200,verbose=20)    
                          
        ll = gbm.best_score
        best_iter = gbm.best_iteration
        cv.append(ll)
        biter.append(best_iter)
        print "---log_loss: %0.6f\n" %ll
        print "---best_iter: %d\n" %best_iter
        gc.collect()
    
    gbm = XGBClassifier(max_depth=8,
                            learning_rate = 0.01,
                            n_estimators=750,
                            subsample=0.6,
                            colsample_bytree=0.3,
                            objective="binary:logistic",
                            silent = False,
                            min_child_weight=1,                       
                            nthread=-1)
                            
    gbm.fit(train[feature_names], np.array(train["target"]),
            eval_metric = "logloss",
            eval_set = [(train[feature_names], np.array(train["target"]))],
                        verbose=20)                            
                        
    tpreds = gbm.predict_proba(test[feature_names])[:, 1]
    df = pd.DataFrame({"ID" : test["ID"], "PredictedProb" : tpreds })
    submission_name = "stacked_v5_xgb.csv"
    df.to_csv(os.path.join(output_dir, submission_name), index = False)
    
    #### calibration
    clf_isotonic = CalibratedClassifierCV(gbm, cv = 10, method = "isotonic")
    start_time = datetime.now()
    clf_isotonic.fit(train[feature_names], np.array(train["target"]))    
    tpreds = clf_isotonic.predict_proba(test[feature_names])[:, 1]
    df = pd.DataFrame({"ID" : test["ID"], "PredictedProb" : tpreds })
    submission_name = "stacked_v3_xgb_calibrated.csv"
    df.to_csv(os.path.join(output_dir, submission_name), index = False)
    end_time = datetime.now()
    print "Duration {0}".format(end_time - start_time)