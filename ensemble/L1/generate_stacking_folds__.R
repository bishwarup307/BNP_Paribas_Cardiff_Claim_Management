# required libraries
require(readr)
require(caret)

# set up directories
# & seed for reproducability
parentDirectory <- "F:/Kaggle/BNP_Paribas/"
rawDataPath <- paste0(parentDirectory, "RawData/")
pipelinePath <- paste0(parentDirectory, "pipeLine4F/")
setwd(parentDirectory)
set.seed(353)

# read the train file
trainPath <- paste0(rawDataPath, "train.csv")
train <- read_csv(trainPath)

# stratified 4 fold
kf <- createFolds(train$target, k = 4, list = TRUE)
maxL <- as.integer(max(sapply(kf, length)))

# initialize the fold frame
fold_ids <- data.frame(tmp = rep(-1, maxL))

for(i in 1:length(kf)){
  row.idx <- kf[[i]]
  ids <- train[row.idx,]$ID
  if(length(ids) < maxL){
    num_nas_needed <- maxL - length(ids)
    ids <- c(ids, rep(NA, num_nas_needed))
  }
  fold_name <- paste0("Fold_", i)
  #print(fold_name)
  fold_ids[[fold_name]] <- ids
}
# delete the temp column
fold_ids$tmp <- NULL

# save the folds to disk
write_csv(fold_ids, paste0(pipelinePath, "fold4f.csv"))