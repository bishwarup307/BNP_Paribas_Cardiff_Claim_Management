require(readr)
require(xgboost)

dir <- "F:/Kaggle/BNP_Paribas/pipeLine4F/"
data_dir <- "./Data/"
eval_dir <- "./evals/"
test_dir <- "./tests/"

setwd(dir)
set.seed(121)

mergeAll <- function(id_, path, key = "ID"){
    merged <- data.frame(ID = id_)
    fileList <- list.files(path)
    for (fileName in fileList){
        predictionFile <- paste0(path, fileName)
        pred <- read_csv(predictionFile)
        pred <- pred[, !names(pred) %in% c("target", "Fold", "ground_truth")]
        merged <- merge(merged, pred)
    }
    cat("\nSuccessfully merged ", length(fileList), "predictions ...")
    return(merged)
}

# read L0
trainFile <- paste0(data_dir, "train_processed_v19.csv")
testFile <- paste0(data_dir, "test_processed_v19.csv")
train <- read_csv(trainFile)
#test <- read_csv(testFile)

# read tsne files
tsne.tr <- read_csv(paste0(data_dir, "tsne_train.csv"))
#tsne.te <- read_csv(paste0(data_dir, "tsne_test.csv"))

# read L1
train_preds <- mergeAll(train$ID, eval_dir)
#tesr_preds <- mergeAll(test$ID, test_dir)

# merge L0 and L1 & tsne features
train <- merge(train, train_preds, by = "ID", all.x = T)
train <- merge(train, tsne.tr, by = "ID", all.x = T)

feature.names <- names(train)[!names(train) %in% c("ID", "target", "train_flag")]

param <- list(objective = "binary:logistic",
              max_depth = 8,
              eta = 0.008,
              subsample = 0.7,
              colsample_bytree = 0.34,
              min_child_weight = 1,
              eval_metric = "logloss")

dtrain <- xgb.DMatrix(data = data.matrix(train[, feature.names]), label = train$target)
watchlist <- list(train = dtrain)
clf <- xgb.train(params = param,
                 data = dtrain,
                 nround = 400,
                 print.every.n = 50,
                 watchlist = watchlist)

imp <- xgb.importance(feature.names, model = clf)
write_csv(imp, paste0(data_dir, "stacked_feature_importance.csv"))