
library(MRSea)
library(car)
library(glmnet)
library(randomForest)
library(data.table)
library(h2o) 
h2o.init(nthreads=-1,max_mem_size = '8G') 

set.seed(7)
train <- fread("C:/Users/Alberto/Google Drive/St Andrews/DM/Practical 2/train.csv", header = T, stringsAsFactors = T)
test <- fread('C:/Users/Alberto/Google Drive/St Andrews/DM/Practical 2/test.csv', header=TRUE, stringsAsFactors = T)
ID <- test$ID

# Delete ID column in training dataset
train[, ID := NULL]

# Delete ID column in testing dataset
test[, ID := NULL]

col_names <- names(train)[c(3:ncol(train))]
train <- na.roughfix(train)
test <- na.roughfix(test)


cate.train <- data.frame(NA)
cate.test <- data.frame(NA)
fact.names <- NULL

for (col in col_names) {
  if (class(train[[col]])=="factor") {
    fact.names <- rbind(col, fact.names)
    levels <- unique(c(train[[col]], test[[col]]))
    cate.train <- cbind(as.integer(train[[col]]), cate.train)
    cate.test <- cbind(as.integer(test[[col]]), cate.test)
    train[[col]] <- NULL
    test[[col]]  <- NULL
  }
}

names(cate.train) <- fact.names
names(cate.test) <- fact.names

# ___________________________________________________________________________________________________________ #

# Training model 1: CATEGORICAL

cate.tot <- cbind(target= as.factor(train$target), cate.train[,-20])
cate.tot <- as.h2o(cate.tot, "cate_tot.hex")

boost <- h2o.gbm(fact.names, "target", cate.tot, distribution = "bernoulli", ntrees= 2000, max_depth = 4, 
                 nfolds =3, learn_rate = 0.1, ignore_const_cols= TRUE)


# Training model 2: QUANTITATIVE
# First: fit to a subset to discover important covariates


fonda <- glm(target~.-v129, train[1:3000,], family = "binomial")
alias(fonda)
vif(fonda)

train.h2o <- as.h2o(train, "tr_tot_ain.hex")
ridge <- h2o.glm(seq(2:ncol(train)), "target", train.h2o, lambda = 0.01696, nfolds = 3,
                 alpha = 0.4, family = "binomial", link = "logit", lambda_search = F)
summary(ridge)

# Log-loss | lambda | alpha
# 0.5152 | 0.007445 | 1
# 0.5151 | 0.00827 | 0.90
# 0.5149 | 0.009307 | 0.8
# 0.5149 | 0.009307 | 0.7
# 0.5147 | 0.01241 | 0.6
# 0.5145 | 0.01696 | 0.4


############################################################################################################

boost <- h2o.gbm(fact.names, "target", cate.tot, distribution = "bernoulli", ntrees= 2000, max_depth = 4, 
                 learn_rate = 0.1, ignore_const_cols= TRUE)
ridge <- h2o.glm(seq(2:ncol(train)), "target", train.h2o, lambda = 0.01696,
                 alpha = 0.4, family = "binomial", link = "logit", lambda_search = F)

boost.fit <- h2o.predict(boost, cate.tot)
ridge.fit <- h2o.predict(ridge, train.h2o[,-1])

boost.fit2 <- as.data.frame(boost.fit)
ridge.fit2 <- as.data.frame(ridge.fit)

featured <- cbind(train$target, boost.fit2["p1"], ridge.fit2["p1"]) # Intermidiate layer
names(featured) <- c("response", "cat", "num")
featured.h2o <- as.h2o(featured, "feat_h2o.hex")

#__________________________________________________________________________________________________________#
# Fit the outer layer

end <- glm(response~cat, "binomial", featured)
termplot(end, partial.resid = T, se= T)

# Fit splines with SALSA1D

splineParams <- makesplineParams(data = featured, varlist = c("num"))
featured$foldid <- getCVids(delete, folds = 3, block = NULL)

salsa1dlist <- list(fitnessMeasure= "AIC", minKnots_1d= c(1), maxKnots_1d= c(3), startKnots_1d= c(2),
                    degree= c(2), maxIterations= 10, gaps= c(0))
salsa1dOutput <- runSALSA1D_withremoval(end, salsa1dlist, varlist = c("num"), splineParams = splineParams,
                                        datain = featured, removal = T)

num.par <- salsa1dOutput$splineParams
cat.par <- salsa1dOutput$splineParams


lafine <- glm(response~bs(cat, degree= 2, knots= c(0.7144487, 0.9152513, 0.9821634))+bs(num, degree= 2, 
              knots= c(0.6617924,0.7083741,0.9348230)), data = featured, family = "binomial")

#___________________________________________________________________________________________________________à
# Get predictions

cate.test.tot <- cate.test[,-20]
cat.test.h2o <- as.h2o(cate.test.tot, "t_h20_est.hex")
num.test.h2o <- as.h2o(test, "tes_h2o_t.hex")

cat.p <- h2o.predict(boost, cat.test.h2o)
num.p <- h2o.predict(ridge, num.test.h2o)
cat.p <- as.data.frame(cat.p)
num.p <- as.data.frame(num.p)

featured.t <- cbind(cat.p["p1"], num.p["p1"]) # Intermidiate layer
names(featured.t) <- c("cat", "num")

final <- predict(lafine, featured.t, type = "response")
tosub <- cbind(ID, PredictedProb= final)
write.csv(tosub, file = "answer_weird.csv", row.names = F)
getwd()




