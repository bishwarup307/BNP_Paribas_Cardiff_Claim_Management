# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:51:51 2016

@author: bishwarup
"""

import os
import shutil
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from ml_metrics import log_loss

# set up directories
parent_dir = "/home/bishwarup/Kaggle/BNP/pipeLine4F/PartitionsVW/"
converter = parent_dir + "csv2vw.py"
num_partitions = 4

# convert csv to vw with phraug2
def convert_to_vw():
    # 4 data partitions    
    for i in xrange(num_partitions):
        
        csv_files_path = "P" + `i+1`+"/"
        in_path = os.path.join(parent_dir, csv_files_path)
        vw_files_path = "P" + `i+1`+"_VW/"
        out_path = os.path.join(parent_dir, vw_files_path)
        
        print "\nFiles in {0} :...".format(csv_files_path)
        file_list = []
        for file_ in os.listdir(in_path):
            if file_.endswith(".csv"):
                file_list.append(file_)
        
        for file_ in file_list:
            
            print "Converting {0} to vowpal_wabbit format...".format(file_)
            in_file = os.path.join(in_path, file_)
            shutil.copy(in_file, out_path)
            in_file = os.path.join(out_path, file_)
            out_file  = os.path.join(out_path, os.path.basename(in_file).split(".")[0] + ".vw")
            cmd = "python " + converter + " -s -l 1 -i 0 -z " + in_file + " " + out_file
            os.system(cmd)
            os.remove(in_file)

# Execute vw for a parituclar fold/ test data
def execute_vw_LR(data_version, fold = 99 ,num_iter = 20, loss = "logistic", learn_rate = 0.7, hash_all = False, validate = True):
    
    # derive train_file names from the data version and fold and is_test
    if validate:
        in_file = os.path.join(parent_dir, ("P" + `data_version`+"_VW/")) + "P" + `data_version` + "_Fold_" + `fold` + "_train.vw"
    else:
        in_file = os.path.join(parent_dir, ("P" + `data_version`+"_VW/")) + "P" + `data_version` + "_train_all.vw"
    
    # generic model name for all the folds
    # we shall delete the model after predictions are generate for each successive folds
    model_path = os.path.join(parent_dir, "vw_models/")
    model_name = model_path + "P" + `data_version` + ".vw"
    
    print "##### data_file : {0} | fold : {1} | model_name : {2} ####".format(os.path.basename(in_file), fold, os.path.basename(model_name))
    # execute on terminal
    if hash_all:
        vw_cmd = "vw -d " + in_file + " -f " + model_name + " --binary --loss_function " + loss + " --passes " + `num_iter` + " -l " + `learn_rate` +" -c -k " + " -q nn " + " --hash all " + " --holdout_off"
        
    else:
        vw_cmd = "vw -d " + in_file + " -f " + model_name + " --binary --loss_function " + loss + " --passes " + `num_iter` + " -l " + `learn_rate` +" -c -k " + " -q nn " + " --holdout_off"
        
    os.system(vw_cmd)
    print "model created..."

# predict on the validation/ test data using the model name from execute_vw_LR
# will generate probabilities using --link logistic
def predict_vw(data_version, fold = 99, is_test = False):
    
    model_path = os.path.join(parent_dir, "vw_models/")
    model_name = model_path + "P" +`data_version` + ".vw"
    
    # derive the test/ validation file names from the data version and fold and is_test
    if is_test:
        test_file = os.path.join(parent_dir, ("P" + `data_version`+"_VW/")) + "P" + `data_version` + "_test.vw"
        print "predicting test ...".format(fold)
    else:
        test_file = os.path.join(parent_dir, ("P" + `data_version`+"_VW/")) + "P" + `data_version` + "_Fold_" + `fold` + "_valid.vw"
        print "predicting fold {0} ...".format(fold)

    # save the predictions in a temp file named "temp_prediction.txt"
    out_file = os.path.join(parent_dir, "Predictions/tmp/") + "temp_prediction.txt"
    vw_cmd = "vw -d " + test_file + " -t -i " + model_name + " -p " + out_file +" --link logistic"
    os.system(vw_cmd)
    
    # now remove the model and cache file from the modeling directory
    model_path = os.path.join(parent_dir, "vw_models/")
    for file_ in os.listdir(model_path):
        to_remove = os.path.join(model_path, file_)
        os.remove(to_remove)
    
# format the prediction txt -> prediction.csv
def create_prediction_file(data_version, fold = 99, is_test = False):

    # we can save the test prediction directly to the test predictions directory    
    if is_test:
        data_file = os.path.join(parent_dir, ("P" + `data_version`+"/")) +"P" + `data_version` + "_test.csv"
        out_file = os.path.join(parent_dir, "Predictions/tests/") + "P" + `data_version` + "_test.csv"
        
    # got to save individual fold predictions to a temp directory names fold_predictions in order to
    # merge them in a single file to be used at second layer
    else:
        data_file = os.path.join(parent_dir, ("P" + `data_version`+"/")) + "P" + `data_version` + "_Fold_" + `fold` + "_valid.csv"
        out_file = os.path.join(parent_dir, "Predictions/fold_predictions/") + "P" + `data_version` + "_Fold_" + `fold` + "_eval.csv"

    # the raw prediction file generated by VW
    text_file_name = os.path.join(parent_dir, "Predictions/tmp/") + "temp_prediction.txt"
    model_version = "VW_P" + `data_version`
    
    # need to match the ID field with the predictions
    with open(out_file, "wb") as prediction_csv:
        prediction_csv.write("raw\n")
        for line in open(text_file_name):
            row = line.strip().split(" ")
            prediction_csv.write("%s\n" % row[0])
            
    if is_test:
        id_df = pd.read_csv(data_file)[["ID"]]
        preds_df = pd.read_csv(out_file)
        assert (id_df.shape[0] == preds_df.shape[0]), "data file and prediction file has differring number of rows..."
        id_df[model_version] = np.array(preds_df["raw"])
        id_df.to_csv(out_file, index = False)
    else:
        id_df = pd.read_csv(data_file)[["ID", "target"]]
        id_df["Fold"] = np.repeat(fold, id_df.shape[0])
        preds_df = pd.read_csv(out_file)
        assert (id_df.shape[0] == preds_df.shape[0]), "data file and prediction file has differring number of rows..."
        id_df[model_version] = np.array(preds_df["raw"])
        id_df.to_csv(out_file, index = False)
        ll = log_loss(np.array(id_df["target"]), np.array(id_df[model_version]))
        print "******************************************************************************"
        print "***** data version : {0} | fold : {1} | fold sample: {2} | log loss {3} ******".format(data_version, fold, id_df.shape[0], np.round(ll, 7))
        print "******************************************************************************"
    os.remove(text_file_name)
        
def merge_folds(data_version):
    
    fold_path = os.path.join(parent_dir, "Predictions/fold_predictions/")
    merged_file_name = "VW_P" + `data_version`+ "_eval.csv"
    train_prediction_path = os.path.join(parent_dir, "Predictions/evals/")
    train_prediction_file = os.path.join(train_prediction_path, merged_file_name)
    
    all_folds = glob.glob(fold_path + "/*.csv")
    eval_matrix = pd.DataFrame()
    fold_list = []
    
    for file_ in all_folds:
        df = pd.read_csv(file_, header = 0)
        fold_list.append(df)
        
    eval_matrix = pd.concat(fold_list)
    assert (eval_matrix.shape[0] == 114321), "Train shape does not match!"
    print "\nSaving combined predictions to {0}".format(os.path.basename(train_prediction_file)) 
    eval_matrix.to_csv(train_prediction_file, index = False)
    
def delete_folds():
    
    fold_path = os.path.join(parent_dir, "Predictions/fold_predictions/")
    print "deleting {0} individual fold predictions...".format(len(os.listdir(fold_path)))
    for files_ in os.listdir(fold_path):
        f = os.path.join(fold_path, files_)
        os.remove(f)
    
if __name__ == '__main__':

    start_time = datetime.now()    
    #convert_to_vw()    
    for i in xrange(1, num_partitions):
        
        data_version = i+1
        print "Starting VW pipeline for partitions P{0} ...".format(data_version)
        
        for j in xrange(4):
            num_fold = j + 1
            execute_vw_LR(data_version= data_version, fold=num_fold, validate=True, num_iter=40,learn_rate=1.1)
            predict_vw(data_version=data_version, fold=num_fold, is_test= False)
            create_prediction_file(data_version=data_version, fold=num_fold, is_test= False)
            
        merge_folds(data_version=data_version)
        execute_vw_LR(data_version=data_version, validate=False, num_iter=40,learn_rate=1.1)   
        predict_vw(data_version=data_version, is_test=True)
        create_prediction_file(data_version=data_version, is_test=True)
        delete_folds()
        
    end_time = datetime.now()
    print "Execution time {0}".format(end_time - start_time)
