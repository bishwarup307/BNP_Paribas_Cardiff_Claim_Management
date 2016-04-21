# ####train on master ##
# total features: 3457 #
########################

# load libraries
require(readr)
require(xgboost)

# set up directory
# set seed for reproducability
dir <- "F:/Kaggle/BNP_Paribas/pipeLine4F/"
eval_dir <- "./evals/"
test_dir <- "./tests/"
setwd(dir)
set.seed(33)

# read the master data files
train <- read_csv("./Data/train_commit_v9_4f.csv")
test <- read_csv("./Data/test_commit_v9_4f.csv")
folds <- read_csv("fold4f.csv")
feature.names <- names(train)[!names(train) %in% c("ID", "target", "train_flag")]

# master meta containers
evalMatrix <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), xgb_master1 = numeric())
testMatrix <- data.frame(ID = test$ID)

cv <- c()
param <- list(objective = "binary:logistic",
              max_depth = 8,
              eta = 0.01,
              subsample = 0.9,
              colsample_bytree = 0.45,
              min_child_weight = 1,
              eval_metric = "logloss")
start_time <- Sys.time()
for(i in 1:4) {
  
  cat("\n---------------------------")
  cat("\n------- Fold: ", i, "----------")
  cat("\n---------------------------\n")
  idx <- folds[[i]]
  idx <- idx[!is.na(idx)]
  
  trainingSet <- train[!train$ID %in% idx,]
  validationSet <- train[train$ID %in% idx,]
  
  cat("\nnrow train: ", nrow(trainingSet))
  cat("\nnrow eval: ", nrow(validationSet), "\n")
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainingSet[, feature.names]), label = trainingSet$target)
  dval <- xgb.DMatrix(data = data.matrix(validationSet[, feature.names]), label = validationSet$target)
  
  
  watchlist <- list(OOB = dval, train = dtrain)
  
  clf <- xgb.train(params = param,
                   data = dtrain,
                   nround = 2130,
                   print.every.n = 50,
                   watchlist = watchlist)
  
  preds <- predict(clf, dval)
  df <- data.frame(Fold = rep(i, nrow(validationSet)), ID = validationSet$ID, ground_truth = validationSet$target, xgb_master1 = preds)
  evalMatrix <- rbind(evalMatrix, df)
  bestScore <- clf$bestScore
  bestIter <- clf$bestInd
  
  cv <- c(cv, bestScore)
  
  cat("\n*** fold score: ", bestScore)
  cat("\n*** best iter: ", bestIter)
  rm(clf, df, trainingSet, validationSet, watchlist)
  gc()
}

dtrain <- xgb.DMatrix(data.matrix(train[, feature.names]), label = train$target)
watchlist <- list(train = dtrain)
clf <- xgb.train(params = param,
                 data = dtrain,
                 nround = 2130,
                 print.every.n = 50,
                 watchlist = watchlist)

tpreds <- predict(clf, data.matrix(test[,feature.names]))
testMatrix[["xgb_master1"]] <- tpreds

# save the predictions to disk
write_csv(evalMatrix, paste0(eval_dir, "xgb_master1_eval.csv"))
write_csv(testMatrix, paste0(test_dir, "xgb_master1_test.csv"))
end_time <- Sys.time()
print(total_time <- end_time - start_time)