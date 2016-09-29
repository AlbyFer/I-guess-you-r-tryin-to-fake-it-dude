library(magrittr)
library(data.table)
library(xgboost)
library(methods)
library(randomForest)
library(ggplot2)
library(Ckmeans.1d.dp)
library(DiagrammeR)
library(caret)

setwd("C:/Users/Alberto/Google Drive/St Andrews/DM/Practical 2")

set.seed(7)
train <- fread('train.csv', header = T, stringsAsFactors = T)
test <- fread('test.csv', header=TRUE, stringsAsFactors = T)
ID <- test$ID

# Delete ID column in training dataset
train[, ID := NULL]

# Delete ID column in testing dataset
test[, ID := NULL]

col_names <- names(train)[c(3:ncol(train))]

# substitute categorical variables with integers
for (col in col_names) {
  if (class(train[[col]])=="factor") {
    levels <- unique(c(train[[col]], test[[col]]))
    train[[col]] <- as.integer(train[[col]])
    test[[col]]  <- as.integer(test[[col]])
  }
}









# Impute the NAs through simple median
train <- na.roughfix(train)
test <- na.roughfix(test)

# Save the name of the last column
name_target <- names(train)[1]

target <- train[, name_target, with = F][[1]]

train[, name_target:=NULL, with = F]

produce_model <- function(train, test, target, file_name, eta) {
  trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
  testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix
  
  
  param <- list("objective" = "binary:logistic",
                "eval_metric" = "logloss", "eta"= eta)
  
  cv.nround <- 5
  cv.nfold <- 3
  
  bst.cv <- xgb.cv(param=param, data = trainMatrix, label = target, 
                  nfold = cv.nfold, nrounds = cv.nround)
  nround <- 50
  bst <- xgboost(param=param, data = trainMatrix, label = target, nrounds=nround)
  pred <- predict(bst, data.matrix(test))
  
  pred <- as.data.frame(pred)
 # ID <- as.data.frame(ID)
  # out <- cbind(ID, "PredictedProb"=pred$pred)
#  write.csv(out, file = file_name, row.names = F)
  
   return(pred)
#  return(bst)
}

bst <- produce_model(train, train, target, "answer_xgb_imp.csv", 0.1) 

names <- dimnames(trainMatrix)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])


train2 <- as.data.frame(c(target, train))
n <- names(train2)
f <- as.formula(paste("target ~", paste(n[!n %in% "target"], collapse = " + ")))
logi <- glm(target~., family = binomial, data = train2)
logi.pred <- predict(logi, as.data.frame(test))

########################################################################################################à

a <- as.data.frame(fread("C:/Users/Alberto/Desktop/answer/answer_xgb_imp.csv"))
b <- as.data.frame(fread("C:/Users/Alberto/Desktop/answer/answer_H2O-deepLearning-e100-bmodel.csv"))
c <- as.data.frame(fread("C:/Users/Alberto/Desktop/answer/answer_boostH2O.csv"))
d <- as.data.frame(fread("C:/Users/Alberto/Desktop/answer/pre.csv"))

ID <- a[,1]
a <- a[, -1]
b <- b[, -1]
c <- c[, -1]
d <- d[,-1]

k <- cbind(a,b,c,d)
cor(k, method = "pearson")

ans <- apply(cbind(a,b), 1, mean)
ans <- cbind(ID, PredictedProb=ans)

write.csv(ans, "final.csv", row.names = F)
getwd()



  