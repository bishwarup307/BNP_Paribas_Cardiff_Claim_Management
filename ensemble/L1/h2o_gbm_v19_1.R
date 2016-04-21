# load libraries
require(readr)
require(h2o)
require(Metrics)

# set up directories
dir <- "F:/Kaggle/BNP_Paribas/pipeLine4F/"
data_dir <- "./Data/"
eval_dir <- "./evals/"
test_dir <- "./tests/"
setwd(dir)
set.seed(153)
pred_name <- "h2o_gbm_v19_1"


#read the data files
cat("\nReading data files...")
trainFile <- paste0(data_dir, "train_processed_v19.csv")
testFile <- paste0(data_dir, "test_processed_v19.csv")
train <- read_csv(trainFile)
test <- read_csv(testFile)
folds <- read_csv("fold4f.csv")
train$target <- as.factor(train$target)

feature.names <- names(train)[!names(train) %in% c("ID", "target", "train_flag")]

#initiate h2o server
h2o.init(nthreads = -1, max_mem_size = '16g')

# train-test meta containers
evalMatrix <- data.frame(Fold = numeric(), ID = numeric(), ground_truth = numeric())
evalMatrix[[pred_name]] <- numeric()
testMatrix <- data.frame(ID = test$ID)

#train[is.na(train)] <- -1
#test[is.na(test)] <- -1

start_time <- Sys.time()
for (ii in 1:ncol(folds)){
  cat("\n----- Fold : ", ii, "-------")
  
  ids <- folds[[ii]]
  trainingSet <- train[!train$ID %in% ids,]
  validationSet <- train[train$ID %in% ids,]
  
  trainHex <- as.h2o(trainingSet)
  validHex <- as.h2o(validationSet)
  
  clf <- h2o.gbm(x = feature.names,
                 y = 'target',
                 training_frame = trainHex,
                 distribution = "bernoulli",
                 ntrees = 2500,
                 max_depth = 7,
                 min_rows = 8,
                 learn_rate = 0.02,
                 sample_rate = 0.9,
                 col_sample_rate = 0.35,
                 validation_frame = validHex,
                 stopping_metric = "logloss")
  
  preds <- as.numeric(as.data.frame(predict(clf, validHex))$p1)
  ll <- logLoss(as.integer(validationSet$target) - 1, preds)
  cat("\n Fold: ", ii, "| logloss: ", ll)
  df <- data.frame(Fold = rep(ii, nrow(validationSet)), ID = validationSet$ID, ground_truth = validationSet$target, pred_name = preds)
  evalMatrix <- rbind(evalMatrix, df)
  rm(clf, trainHex, validHex, trainingSet, validationSet, preds)
  
}

trainHex <- as.h2o(train)
test$target <- NULL
testHex <- as.h2o(test)
clf <- h2o.gbm(x = feature.names,
               y = 'target',
               training_frame = trainHex,
               distribution = "bernoulli",
               ntrees = 2500,
               max_depth = 7,
               min_rows = 8,
               learn_rate = 0.02,
               sample_rate = 0.9,
               col_sample_rate = 0.35,
               stopping_metric = "logloss")

tpreds <- as.numeric(as.data.frame(predict(clf, testHex))$p1)
testMatrix[[pred_name]] <- tpreds

# save predictions to disk
eval_predict_file <- paste0(eval_dir, pred_name, "_eval.csv")
test_predict_file <- paste0(test_dir, pred_name, "_test.csv")
cat("\nWriting train set predictions in ", eval_predict_file)
write_csv(evalMatrix, eval_predict_file)
cat("\nWriting test set predictions in ", test_predict_file)
write_csv(testMatrix, test_predict_file)

end_time <- Sys.time()
print(time_elpased <- end_time - start_time)
