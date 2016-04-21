# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:36:06 2016

@author: bishwarup
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.optimizers import Adagrad,SGD,Adadelta
from keras.regularizers import l1, l2, l1l2
from keras.callbacks import Callback

parent_dir = "/home/bishwarup/Kaggle/BNP/pipeLine4F/"
data_dir = os.path.join(parent_dir, "Data/")
eval_dir = os.path.join(parent_dir, "evals/")
test_dir = os.path.join(parent_dir, "tests/")
train_file = os.path.join(data_dir, "train_processed_v19.csv")
test_file = os.path.join(data_dir, "test_processed_v19.csv")
fold_file = os.path.join(parent_dir, "fold4f.csv")
pred_name = "nn_v19_1"
eval_preds_file = os.path.join(eval_dir, pred_name + "_eval.csv")
test_preds_file = os.path.join(test_dir, pred_name + "_test.csv")

nb_epoch = 150
batch_size = 8
n_class = 2

def load_data():
    
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    fold_ids = pd.read_csv(fold_file)
    
    print "train shape {}".format(train.shape)
    print "test shape {}".format(test.shape)
    
    alldata = pd.concat([train, test], axis  = 0, ignore_index = True)
    
    use_cols = [f for f in alldata.columns if f not in["ID", "target", "train_flag"]]
    use_df = alldata[use_cols].copy()
    use_df.fillna(-1, inplace = True)
    scalar = StandardScaler()
    use_df = pd.DataFrame(scalar.fit_transform(use_df))
    use_df.columns = use_cols
    
    alldata.drop(use_cols, axis = 1, inplace = True)
    assert (alldata.shape[1] == 3), "data shape mismatch!"
    alldata = pd.concat([alldata, use_df], axis = 1)
    del use_cols, use_df, scalar
    
    train_proc = alldata[alldata.train_flag == 1].copy()
    test_proc = alldata[alldata.train_flag == 0].copy().reset_index(drop = True)
    del alldata
    
    return train_proc, test_proc, fold_ids
    
def build_model(train):

    input_dim = train.shape[1]
    classes = 2    
    
    model = Sequential()
    model.add(Dropout(0.1, input_shape = (input_dim,)))
    model.add(Dense(30, init = "glorot_uniform", W_regularizer=l2(1e-6)))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.4))

    '''
    model.add(Dense(500, init = 'glorot_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.9))
    '''
    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer="adam")
    return model  
    
def fit_model(ptr, pte, fold_ids, model_version):
    
    feature_names = [f for f in ptr.columns if f not in ["ID", "target", "train_flag"]]

    print "\nStarting validation...."

    eval_matrix = pd.DataFrame(columns = ["Fold", "ID", "ground_truth", model_version])        
    test_matrix = pd.DataFrame({"ID" : pte["ID"]})
    
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
        model.fit(tr_X, tr_Y, nb_epoch=nb_epoch, batch_size= batch_size, validation_data= (val_X, val_Y))
        
        preds = model.predict_proba(val_X, batch_size=128)[:, 1]
        df = pd.DataFrame({"Fold" : np.repeat((i + 1), validationSet.shape[0]) ,"ID" : validationSet["ID"], "ground_truth" : validationSet["target"], 
                            model_version : preds})
        eval_matrix = eval_matrix.append(df, ignore_index = True)
        del model, tr_X, tr_Y, val_X, val_Y
     
     
    tr_X = np.matrix(ptr[feature_names])
    tr_Y = np_utils.to_categorical(np.array(ptr["target"].copy()))
    te_X = np.matrix(pte[feature_names])

    model = build_model(tr_X)
    model.fit(tr_X, tr_Y, nb_epoch=nb_epoch, batch_size= batch_size)
    tpreds =   model.predict_proba(te_X, batch_size=128)[:, 1]
    test_matrix[model_version] = tpreds
    return eval_matrix, test_matrix
    del model, tr_X, tr_Y, te_X
    gc.collect()


if __name__ == '__main__':
    
    print "reading and processing data files..."
    train, test, fold_ids = load_data()   
    
    train_bag = np.zeros((train.shape[0], 4))
    test_bag = np.zeros((test.shape[0], 2))
    
    seed_list = [322, 1136, 458, 957, 114, 369, 475, 1234, 470, 11]
    
    for j in xrange(len(seed_list)):
        print "################################################"
        print "############### Starting bag {0}  ################".format(j+1)
        print "################################################"
        np.random.seed(seed_list[j])
        eval_mat, test_mat = fit_model(train, test, fold_ids, pred_name)   # 4-fold cross validation         
        eval_mat, test_mat = np.matrix(eval_mat), np.matrix(test_mat)
        train_bag += eval_mat
        test_bag += test_mat
        gc.collect()
    
    train_bag /= len(seed_list)
    test_bag /= len(seed_list)
        
    train_bag, test_bag = pd.DataFrame(train_bag), pd.DataFrame(test_bag)
    train_bag.columns = ["Fold", "ID", "ground_truth", pred_name]
    train_bag["Fold"] = train_bag["Fold"].map(lambda x: x.astype(int))
    train_bag["ID"] = train_bag["ID"].map(lambda x: x.astype(int))
    train_bag["ground_truth"] = train_bag["ground_truth"].map(lambda x: x.astype(int))
    test_bag.columns = ["ID", pred_name]         
    
    print "saving train predictions to {} ...".format(os.path.basename(eval_preds_file))
    train_bag.to_csv(eval_preds_file, index = False)
    print "saving test predictions to {} ...".format(os.path.basename(test_preds_file))
    test_bag.to_csv(test_preds_file, index = False)
    

    