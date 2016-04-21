# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 01:13:21 2016

@author: bishwarup
"""

from __future__ import division
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import theano
theano.config.openmp = True
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.regularizers import l1, l2, l1l2

# declare global variables
parent_dir = "/home/bishwarup/Kaggle/BNP/"
child_dir = os.path.join(parent_dir, 'pipeline4F/')
nb_epoch = 120
nb_batch_size = 8
        
# Keras model building block
def build_model(train):

    input_dim = train.shape[1]
    classes = 2    
    
    model = Sequential()
    model.add(Dropout(0.1, input_shape = (input_dim,)))
    model.add(Dense(30, init = "glorot_uniform", W_regularizer=l1(1e-6)))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))
        
    model.compile(loss='binary_crossentropy', optimizer="adagrad")
    return model    

# cross-validate to generate train meta features
# &
# train on all instances to generate test meta features
def fit_model(ptr, pte, fold_ids, model_version):
    
    feature_names = [f for f in ptr.columns if f not in ["ID", "target", "train_flag"]]

    print "\nStarting validation...."

    eval_matrix = pd.DataFrame(columns = ["Fold", "ID", "ground_truth", model_version])        
    test_matrix = pd.DataFrame({"ID" : pte["ID"]})
    
    # generate train predictions
    for i in xrange(fold_ids.shape[1]):
        
        print "\n--------------------------------------------"
        print "---------------- Fold %d --------------------" %i
        print "--------------------------------------------"
        
        val_ids = fold_ids.ix[:, i].dropna()
        idx = ptr["ID"].isin(list(val_ids))
        trainingSet = ptr[~idx]
        validationSet = ptr[idx]
        tr_Y = np_utils.to_categorical(np.array(trainingSet["target"].copy()))
        val_Y = np_utils.to_categorical(np.array(validationSet["target"].copy()))
        tr_X = np.matrix(trainingSet[feature_names])
        val_X = np.matrix(validationSet[feature_names])
                
        model = build_model(tr_X)   
        model.fit(tr_X, tr_Y, nb_epoch=nb_epoch, batch_size=nb_batch_size, validation_data= (val_X, val_Y))
        
        preds = model.predict_proba(val_X, batch_size=128)[:, 1]
        df = pd.DataFrame({"Fold" : np.repeat((i + 1), validationSet.shape[0]) ,"ID" : validationSet["ID"], "ground_truth" : validationSet["target"], 
                            model_version : preds})
        eval_matrix = eval_matrix.append(df, ignore_index = True)
        del model, tr_X, tr_Y, val_X, val_Y
     
    # generate test predictions
    tr_X = np.matrix(ptr[feature_names])
    tr_Y = np_utils.to_categorical(np.array(ptr["target"].copy()))
    te_X = np.matrix(pte[feature_names])

    model = build_model(tr_X)
    model.fit(tr_X, tr_Y, nb_epoch=nb_epoch, batch_size=nb_batch_size)
    tpreds =   model.predict_proba(te_X, batch_size=128)[:, 1]
    test_matrix[model_version] = tpreds
    return eval_matrix, test_matrix
    
if __name__ == "__main__":
    
    fold_index_path = child_dir + "fold4f.csv"
    fold_ids = pd.read_csv(fold_index_path)
    
    # loop over all the four data partitions
    for i in xrange(4):
        
            
        train_path = child_dir + "Partitions/"  + "trainP" + `i+1` + ".csv"
        test_path = child_dir + "Partitions/"  + "testP" + `i+1` + ".csv"
        model_version = "NN_P" + `i + 1` 
        print "************************************"
        print "Training model version {0}".format(model_version)        
        print "************************************"
        
        print "Reading {0} ....".format(train_path)
        tr = pd.read_csv(train_path)
        print "Reading {0} ....".format(test_path)
        te = pd.read_csv(test_path)
        
        print "merging train and test... "
        tr["train_flag"] = 1
        te["train_flag"] = 0
        te["target"] = -1
        alldata = pd.concat([tr, te], axis = 0, ignore_index = True)
        alldata.fillna(-1, inplace = True)
        
        print "normalizing the features..."
        standardize_cols = [f for f in alldata.columns if f not in ["ID", "target", "train_flag"]]
        standardize_df = alldata[standardize_cols]
        
        scalar = StandardScaler()
        #scalar = MinMaxScaler()
        standardize_df = pd.DataFrame(scalar.fit_transform(standardize_df))
        standardize_df.columns = standardize_cols
        
        alldata.drop(standardize_cols, axis = 1, inplace = True)
        alldata = pd.concat([alldata, standardize_df], axis = 1)
        
        train = alldata[alldata["train_flag"] == 1].copy().reset_index(drop  = True)
        test = alldata[alldata["train_flag"] == 0].copy().reset_index(drop = True)
        
        seed_list = [12, 136, 458, 957, 140, 369, 475, 1234, 470, 5555]
        
        train_bag = np.zeros((train.shape[0], 4))
        test_bag = np.zeros((test.shape[0], 2))

        # bag the predictions        
        for j in xrange(len(seed_list)):
            print "################################################"
            print "##### Starting bag {0} model_ver {1} ###########".format(j+1, model_version)
            print "################################################"
            np.random.seed(seed_list[j])
            eval_mat, test_mat = fit_model(train, test, fold_ids, model_version)   # 4-fold cross validation         
            eval_mat, test_mat = np.matrix(eval_mat), np.matrix(test_mat)
            train_bag += eval_mat
            test_bag += test_mat
            
        train_bag /= len(seed_list)
        test_bag /= len(seed_list)
        
        train_bag, test_bag = pd.DataFrame(train_bag), pd.DataFrame(test_bag)
        train_bag.columns = ["Fold", "ID", model_version, "ground_truth"]
        train_bag["Fold"] = train_bag["Fold"].map(lambda x: x.astype(int))
        train_bag["ID"] = train_bag["ID"].map(lambda x: x.astype(int))
        train_bag["ground_truth"] = train_bag["ground_truth"].map(lambda x: x.astype(int))
        test_bag.columns = ["ID", model_version]                
        
        train_prediction_file = child_dir + "evals/" + model_version + "_eval.csv"
        test_prediction_file = child_dir + "tests/" + model_version + "_test.csv"
        
        # save the meta features to disk
        train_bag.to_csv(train_prediction_file, index = False)
        test_bag.to_csv(test_prediction_file, index = False)
        
        del train_bag, test_bag,standardize_df, train, test, tr, te, standardize_cols