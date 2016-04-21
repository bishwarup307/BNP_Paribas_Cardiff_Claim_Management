##################
# Bagged 5 times #
##################

# load libraries
require(readr)
require(Metrics)
require(xgboost)

# set working directory and seed
dir = "F:/Kaggle/BNP_Paribas/"
child_path = "./pipeLine4F/"
test_path = paste0(child_path, "Data/test_commit_v9_4f.csv")
train_prediction_path <- paste0(child_path, "evals/")
test_prediction_path <- paste0(child_path, "tests/")
setwd(dir)

# read the validation IDs for stacking
folds <- read_csv("./pipeLine4F/fold4f.csv")

# seed list for bagging
seed_list <- c(1, 544, 353, 13, 12345)

# number of iterations of xgb for
# 4 feature partitions
# chosen by cross validation
num_rounds <- c(1420, 1460, 1420, 1530)

# xgboost parameters
param <- list(objective = "binary:logistic",
              max_depth = 9,
              eta = 0.01,
              subsample = 0.9,
              colsample_bytree = 0.5,
              min_child_weight = 1,
              eval_metric = "logloss")

start_time <- Sys.time()

for(ii in 4:4){
  # loop for partitions  
  cat("** training with partition", ii, " ***\n")
  train_file <- paste(child_path, "./Partitions/trainP", ii, ".csv", sep = "")
  test_file <- paste(child_path, "./Partitions/testP", ii, ".csv", sep = "")
  model_version <- paste("XGB",paste0("train_P", ii), sep = "_")
  
  cat("\nReading ", train_file, "...\n")
  train <- read_csv(train_file)
  cat("\nReading ", test_file, "...\n")
  test <- read_csv(test_file)
  cat("\n** # rows in train: ", nrow(train))
  cat("\n** # rows in test: ", nrow(test))
  feature.names <- names(train)[!names(train) %in% c("ID", "target", "train_flag")]
  
  # initiate empty train and test bag predictions
  train_bag <- matrix(rep(0, 4*nrow(train)), ncol= 4)
  test_bag <- matrix(rep(0, 2*nrow(test)), ncol = 2)
  for(bb in 1:length(seed_list)){
    #loop for bagging
    cat("\n**Staring bag: ", bb, "**\n")
    set.seed(seed_list[bb])
    
    # train and test meta containers
    eval_matrix <- data.frame(ID = numeric(), Fold = numeric(), ground_truth = numeric())
    eval_matrix[model_version] = numeric()
    test_matrix <- data.frame(ID = test$ID)
    
    # test DMATRIX
    dtest <- xgb.DMatrix(as.matrix(test[, feature.names]))
    
    for(jj in 1:length(folds)){
      #loop for validation
      cat("\n---------------------------")
      cat("\n------- Fold: ", jj, "----------")
      cat("\n---------------------------\n")
      
      idx <- folds[[jj]]
      idx <- idx[!is.na(idx)]
      trainingSet <- train[!train$ID %in% idx,]
      validationSet <- train[train$ID %in% idx,]
      
      dtrain <- xgb.DMatrix(data = data.matrix(trainingSet[, feature.names]), label = trainingSet$target)
      dval <- xgb.DMatrix(data = data.matrix(validationSet[, feature.names]), label = validationSet$target)
      
      watchlist <- list(OOB = dval, train = dtrain)
      
      bst <- xgb.train(params = param,
                       data = dtrain,
                       nround = as.integer(num_rounds[ii]),
                       print.every.n = 500,
                       watchlist = watchlist)
      
      fold_preds <- predict(bst, dval)
      cat("\nFold OOB score: ", logLoss(validationSet$target, fold_preds))
      df <- data.frame(ID = validationSet$ID, Fold = rep(jj, nrow(validationSet)), ground_truth = validationSet$target, xgb1_preds = fold_preds)
      eval_matrix <- rbind(eval_matrix, df)
      rm(bst, idx, dtrain, dval)
      gc()
    }
    
    dtrain <- xgb.DMatrix(data= data.matrix(train[, feature.names]), label = train$target)
    watchlist <- list(train = dtrain)
    cat("\nTraining on all data...\n")
    bst <- xgb.train(params = param,
                     data = dtrain,
                     nround = as.integer(num_rounds[ii]),
                     print.every.n = 500,
                     watchlist = watchlist)
    
    tpreds <- predict(bst, dtest)
    test_matrix[model_version] = tpreds
    
    train_bag <- train_bag + as.matrix(eval_matrix)
    test_bag <- test_bag + as.matrix(test_matrix)
    rm(bst, eval_matrix, test_matrix)
    gc()
  }

  train_bag <- train_bag/length(seed_list)
  test_bag <- test_bag/length(seed_list)
  
  train_bag <- as.data.frame(train_bag)
  test_bag <- as.data.frame(test_bag)
  names(train_bag) <- c("ID", "Fold", "ground_truth", model_version)
  names(test_bag) <- c("ID", model_version)
  
  train_prediction_file <- paste0(train_prediction_path, "./", model_version, "_eval.csv")
  test_prediction_file <- paste0(test_prediction_path, "./",model_version, "_test.csv")
  # save train and test predictions
  write_csv(train_bag, train_prediction_file)
  write_csv(test_bag, test_prediction_file)
  
  rm(train_bag, test_bag)
  gc()
} 

end_time <- Sys.time()
print(total_time <- end_time - start_time)
