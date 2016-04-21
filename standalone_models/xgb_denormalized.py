# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 11:33:41 2016

@author: Bishwarup
"""
from __future__ import division
import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import likelihoodEncoder as lhE
from itertools import combinations
from xgboost import XGBClassifier

#### set directories and seed
parent_dir = "F:/Kaggle/BNP_Paribas"
raw_data_dir = os.path.join(parent_dir, "Data/")
processed_data_dir = os.path.join(parent_dir, "data_v19/")
submission_dir = os.path.join(parent_dir, 'Submissions/')
np.random.seed(11)

# consolidate low-frequent categories of categorical features
def consolidate(df, col, threshold = 20):
    raw = df[col].copy()
    cnt = raw.value_counts()
    low_cardinal = list(cnt[cnt < threshold].index)
    raw[raw.isin(low_cardinal)] = "OTHER"
    return np.array(raw)

# denormalize the continuous features - get rid of random noise
def denormalize(df, col):    
    vals = df[col].copy().dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 1e-5]
    denom = vals.value_counts().idxmax() 
    denormalized = np.round(np.array(df[col]/denom),0).astype(int)
    return denormalized

# factorize integer features
def factorize(df, col):
    factorized = df[col].map(lambda x: col + "_" + str(x))
    return factorized

if __name__ == '__main__':
    
    start_time = datetime.now()
    
    print 'Reading data files...'
    train_file = os.path.join(raw_data_dir, "train.csv")
    test_file = os.path.join(raw_data_dir, "test.csv")
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    fold_ids = pd.read_csv(os.path.join(parent_dir, 'fold4f.csv'))
    train["train_flag"] = 1
    test["train_flag"] = 0
    test["target"] = -1
    alldata = pd.concat([train, test], axis = 0, ignore_index = True)
    
    # number of NAs per row
    nan_calc_cols = [f for f in alldata.columns if f not in ["ID", "train_flag", "target"]]
    alldata["na_count"] = alldata[nan_calc_cols].apply(lambda x: pd.isnull(x).sum(), axis = 1)
    
    # some stats on the numeric features
    cat_cols = [f for f in alldata.columns if alldata[f].dtype == "object"]
    numeric_cols = [f for f in alldata.columns if f not in list(cat_cols + ["ID", "train_flag", "target"]) and len(alldata[f].unique()) > 100]
    alldata["row_sum"]  = alldata[numeric_cols].apply(lambda x: np.nansum(x), axis = 1)
    alldata["row_mean"] = alldata[numeric_cols].apply(lambda x: np.nanmean(x), axis = 1)
    alldata["row_median"] = alldata[numeric_cols].apply(lambda x: np.nanmedian(x), axis = 1)
    alldata["row_sd"] = alldata[numeric_cols].apply(lambda x: np.nanstd(x), axis = 1)
    alldata["row_md"] = (alldata["row_mean"] - alldata["row_median"])/alldata["row_sd"]
    
    # features to denormalize
    denormalize_cols = ["v50", "v12", "v10", "v40", "v34", "v14", "v114", "v21"] 
    denom_cols = []
    for num, f in enumerate(denormalize_cols):
        print "{} | denormalizing {}".format(num + 1, f)
        cname = f + "_denom"
        alldata[cname] = denormalize(alldata, f)
        denom_cols.append(cname)
    
    # features to factorize  
    factorize_cols = denom_cols + ["v38", "v62", "v72", "v129"]
    for num, f in enumerate(factorize_cols):
        print "{} | categorizing {}".format(num + 1, f)
        alldata[f] = factorize(alldata, f)
        
    alldata.drop("v107", axis = 1, inplace = True)
    
    #create two_way interactions
    cat_cols = [f for f in alldata.columns if alldata[f].dtype == "object"]
    combi = list(combinations(cat_cols, 2))
    inter_2ways = []
    for num, (i, j) in enumerate(combi):
        print "{} |generating interactions : {} | {} ...".format(num+1, i, j)
        cname = i + "_" + j
        inter_2ways.append(cname)
        alldata[cname] = alldata[i] + "_" + alldata[j]
        
    # create three way interactions
    three_way = ["v66", "v31", "v47", "v79", "v114_denom",  "v129", "v22", "v56", "v24", "v40_denom", "v50_denom"]
    combi_three_way = list(combinations(three_way, 3))
    inter_3ways = []
    for num, (i, j, k) in enumerate(combi_three_way):
        print "{} |generating interactions : {} | {} | {} ...".format(num+1, i, j, k)
        cname = i + "_" + j + "_" + k
        inter_3ways.append(cname)
        alldata[cname] = alldata[i] + "_" + alldata[j] + "_" + alldata[k]
        
    train_processed = alldata[alldata.train_flag == 1].copy()
    test_processed = alldata[alldata.train_flag == 0].copy().reset_index(drop = True)
    del alldata
    gc.collect()
    
    # likelihood coding of the engineered interaction features
    cat_cols = [f for f in train_processed.columns if train_processed[f].dtype == "object"]
    y = train_processed["target"].copy()
    tr_cat, te_cat = lhE.get_ctr_features(train_processed, test_processed, y, cat_cols, 0.75, 0.5, fold_ids)
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
        
        gbm = XGBClassifier(max_depth=8,
                            learning_rate = 0.01,
                            n_estimators=10000,
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
        gc.collect()
    
    best_i = np.mean(biter) + 50
    # train on whole data
    gbm = XGBClassifier(max_depth=8,
                        learning_rate = 0.01,
                        n_estimators=best_i,
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
    submission_file = os.path.join(submission_dir, "xgb_denormalized.csv")
    sub.to_csv(submission_file, index = False)
    
    end_time = datetime.now()
    print 'elapsed time: {}'.format(end_time - start_time)