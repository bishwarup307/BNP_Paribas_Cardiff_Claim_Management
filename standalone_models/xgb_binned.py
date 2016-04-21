# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:03:39 2016

@author: Bishwarup
"""

from __future__ import division
import os
import numpy as np
import pandas as pd
import code_response_rate as cf
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from ml_metrics import log_loss
from itertools import combinations
import xgboost as xgb

parent_dir = 'F:/Kaggle/BNP_Paribas'
data_dir = os.path.join(parent_dir, 'RawData/')
submission_dir = os.path.join(parent_dir, 'Submissions/')
np.random.seed(11)

# bin the continuous features in 100 tiles
def binner(key, maxbins = 101, na = -100, percent_per_bin = 1):
    
    raw_column = alldata[key].copy()
    raw_column.fillna(na, inplace = True)
    
    akey = raw_column[raw_column != na]
    
    count = len(akey.unique())
    
    if count < maxbins:
        return (alldata[key], None)
    
    try:
        bins = np.unique(np.percentile(akey, np.arange(0, 100, percent_per_bin)))
        # Add a bin for NA
        if np.min(raw_column) == na:
            bins = np.insert(bins, 0, na + 1)
        count = len(bins)
    
        # print(key, count)
    
        binned_column = np.digitize(raw_column, bins)
        binned_column = [key + "_" + str(x) for x in binned_column]
        return (binned_column)
    except:
        return (raw_column)

if __name__ == '__main__':

    # read raw data files...
    print "reading data files..."
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    train["train_flag"] = 1
    test["train_flag"] = 0
    test["target"] = -1
    alldata = pd.concat([train, test], axis = 0, ignore_index = True)
    
    # number of NAs per row
    nan_calc_cols = [f for f in alldata.columns if f not in ["ID", "train_flag", "target"]]
    alldata["na_count"] = alldata[nan_calc_cols].apply(lambda x: pd.isnull(x).sum(), axis = 1)
    
    # transforming a few numeric columns into characters to use in interactions
    alldata["v38"] = alldata["v38"].map(lambda x: "v38_" + str(x))
    alldata["v62"] = alldata["v62"].map(lambda x: "v62_" + str(x))
    alldata["v72"] = alldata["v72"].map(lambda x: "v72_" + str(x))
    alldata["v129"] = alldata["v129"].map(lambda x: "v129_" + str(x))
    alldata["v50_binned"] = binner("v50")
    alldata["v12_binned"] = binner("v12")
    alldata["v40_binned"] = binner("v40")
    alldata["v10_binned"] = binner("v10")
    alldata["v114_binned"] = binner("v114")
    alldata["v21_binned"] = binner("v21")
    alldata["v34_binned"] = binner("v34")
    alldata["v14_binned"] = binner("v14")
    
    cat_cols = [f for f in alldata.columns if alldata[f].dtype == "object"]
    
    # some stats on the numeric features
    numeric_cols = [f for f in alldata.columns if f not in list(cat_cols + ["ID", "train_flag", "target"]) and len(alldata[f].unique()) > 100]
    alldata["row_sum"]  = alldata[numeric_cols].apply(lambda x: np.nansum(x), axis = 1)
    alldata["row_mean"] = alldata[numeric_cols].apply(lambda x: np.nanmean(x), axis = 1)
    alldata["row_median"] = alldata[numeric_cols].apply(lambda x: np.nanmedian(x), axis = 1)
    alldata["row_sd"] = alldata[numeric_cols].apply(lambda x: np.nanstd(x), axis = 1)
    alldata["row_md"] = (alldata["row_mean"] - alldata["row_median"])/alldata["row_sd"]
    
    # two-way interactions
    combi = list(combinations(cat_cols, 2))
    for (i, j) in combi:
        alldata[i + "_" + j] = alldata[i] + "_" + alldata[j]
        
    # three-way interactions
    three_way = ["v50_binned", "v66", "v31", "v47", "v56", "v79", "v12_binned", "v114_binned", "v10_binned", "v22", "v129",
                 "v40_binned", "v21_binned", "v24", "v113", "v14_binned", "v34_binned"]
    combi_three_way = list(combinations(three_way, 3))
    for (i, j, k) in combi_three_way:
        alldata[i + "_" + j + "_" + k] = alldata[i] + "_" + alldata[j] + "_" + alldata[k]
    
    # get all the categorical columns after extracting the interactions
    cat_cols = [f for f in alldata.columns if alldata[f].dtype == "object"]
    
    train_processed = alldata[alldata["train_flag"] == 1].copy()
    test_processed = alldata[alldata["train_flag"] == 0].copy().reset_index(drop = True)
    
    # code the interactions and categorical columns with response rate
    y = train_processed["target"].copy()
    tr_cat, te_cat = cf.get_ctr_features(train_processed, test_processed, y, cat_cols, 0.75, 0.5)
    tr_cat, te_cat = pd.DataFrame(tr_cat), pd.DataFrame(te_cat)
    tr_cat.columns = cat_cols
    te_cat.columns = cat_cols
    
    train_processed.drop(cat_cols, axis = 1, inplace = True)
    test_processed.drop(cat_cols, axis = 1, inplace = True)
    
    train_processed = pd.concat([train_processed, tr_cat], axis = 1)
    test_processed = pd.concat([test_processed, te_cat], axis = 1)
    
    target = train_processed["target"].copy()
    train_processed.drop(["ID", "target", "train_flag"], axis = 1, inplace = True)
    skf = StratifiedKFold(target, n_folds = 5, shuffle = True, random_state = 14)
    
    cv = []
    biter = []
    
    for fold, (itr, icv) in enumerate(skf):
        
        print "------ Fold %d -----------\n" %(fold+1)
        X_train = train_processed.iloc[itr]
        X_valid = train_processed.iloc[icv]
        Y_train = target[itr]
        Y_valid = target[icv]
        
        gbm = xgb.XGBClassifier(max_depth=8,
                            learning_rate = 0.01,
                            n_estimators=3000,
                            subsample=0.9,
                            colsample_bytree=0.45,
                            objective="binary:logistic",
                            silent = False,
                            min_child_weight=1,                       
                            nthread=-1)                    
        gbm.fit(X_train, Y_train,
                eval_metric="logloss",
                eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
                         early_stopping_rounds=200,verbose=20)    
                          
        ll = gbm.best_score
        best_iter = gbm.best_iteration
        cv.append(ll)
        biter.append(best_iter)
        print "---log_loss: %0.6f\n" %ll
        print "---best_iter: %d\n" %best_iter
    
    # train on whole data
    gbm = xgb.XGBClassifier(max_depth=8,
                        learning_rate = 0.01,
                        n_estimators=2170,
                        subsample=0.9,
                        colsample_bytree=0.45,
                        objective="binary:logistic",
                        silent = False,
                        min_child_weight=1,                       
                        nthread=-1)
                            
    gbm.fit(train_processed, target, eval_metric="logloss",
            eval_set = [(train_processed, target)],
                        verbose=20)                        
    
    tid = test_processed["ID"].copy()
    assert (len(tid) == 114393), "test length does not match!"
    test_processed.drop(["ID", "target", "train_flag"], axis = 1, inplace = True)
    tpreds = gbm.predict_proba(test_processed)[:, 1]
    sub = pd.DataFrame({"ID" : tid, "PredictedProb" : tpreds})
    submission_file = os.path.join(submission_dir, 'xgb_commit_9.csv')
    sub.to_csv(submission_file, index = False)     