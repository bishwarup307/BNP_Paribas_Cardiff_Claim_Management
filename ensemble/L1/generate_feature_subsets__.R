# load libraries
require(readr)
require(xgboost)

# set-up directories
dir <- "F:/Kaggle/BNP_Paribas/"
setwd(dir)

print(start_time <- Sys.time())

# read the master data files
train <- read_csv("./pipeLine4F/Data/train_commit_v9_4f.csv")
test <- read_csv("./pipeLine4F/Data/test_commit_v9_4f.csv")

# feature set to partition
feature.names <- names(train)[!names(train) %in% c("ID", "target", "train_flag")]

# we train a xgb to get the
# individual feature importance
# with which we shall create 4
# different partitions of the
# data by rule modulo 4
param <- list(objective = "binary:logistic",
              max_depth = 9,
              eta = 0.01,
              subsample = 0.9,
              colsample_bytree = 0.45,
              min_child_weight = 1,
              eval_metric = "logloss")

dtrain <- xgb.DMatrix(data = data.matrix(train[, feature.names]), label = train$target)
watchlist <- list(train = dtrain)

bst <- xgb.train(data= dtrain,
                 params = param,
                 nrounds = 500,
                 watchlist = watchlist,
                 print.every.n = 100)

imp <- xgb.importance(feature.names, model = bst)

#feature partition based on xgb importance - mod 4
fset1 <- c(); fset2 <- c(); fset3 <- c(); fset4 <- c();
for (i in 1:nrow(imp)){
  if(i %% 4 == 1){
    fset1 <- c(fset1, imp$Feature[i])
  }else if(i %% 4 == 2) {
    fset2 <- c(fset2, imp$Feature[i])
  }else if(i %% 4 == 3){
    fset3 <- c(fset3, imp$Feature[i])
  }else {
    fset4 <- c(fset4, imp$Feature[i])
  }
}
# train partitions
dset_tr1 <- cbind(train[, c("ID", "target")], train[, fset1])
dset_tr2 <- cbind(train[, c("ID", "target")], train[, fset2])
dset_tr3 <- cbind(train[, c("ID", "target")], train[, fset3])
dset_tr4 <- cbind(train[, c("ID", "target")], train[, fset4])
# test partitions
dset_te1 <- cbind(test[, c("ID")], test[, fset1])
dset_te2 <- cbind(test[, c("ID")], test[, fset2])
dset_te3 <- cbind(test[, c("ID")], test[, fset3])
dset_te4 <- cbind(test[, c("ID")], test[, fset4])

#write train partitions
write_csv(dset_tr1, "./pipeLine4F/Partitions/trainP1.csv")
write_csv(dset_tr2, "./pipeLine4F/Partitions/trainP2.csv")
write_csv(dset_tr3, "./pipeLine4F/Partitions/trainP3.csv")
write_csv(dset_tr4, "./pipeLine4F/Partitions/trainP4.csv")
#write test partitions
write_csv(dset_te1, "./pipeLine4F/Partitions/testP1.csv")
write_csv(dset_te2, "./pipeLine4F/Partitions/testP2.csv")
write_csv(dset_te3, "./pipeLine4F/Partitions/testP3.csv")
write_csv(dset_te4, "./pipeLine4F/Partitions/testP4.csv")

print(end_time <- Sys.time())
print(total_time <- end_time - start_time)