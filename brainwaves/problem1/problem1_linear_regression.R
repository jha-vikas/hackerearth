library(caret)
library(data.table)
#library(pROC)
#the following line will create a local 4-node snow cluster
library(doParallel)
library(lubridate)
library(dummies)

workers=makeCluster(detectCores(),type="SOCK")
registerDoParallel(workers)

foreach(i=1:4) %dopar% Sys.getpid()


setwd("C:/Users/vjha3/vikas/misc/hackerearth/brainwaves/problem1/")

d_train <- fread("./input/train.csv", na.strings = c("", " ", "NA"))
d_test <- fread("./input/test.csv", na.strings = c("", " ", "NA"))

#View(d_train)
apply(d_train, 2, FUN = function(x) length(unique(x)))
apply(d_test, 2, FUN = function(x) length(unique(x)))

names(d_train)

q <- apply(d_train, 2, function(x) length(table(is.na(x))))
missing_variables_train <- names(d_train)[q>1]
missing_variables_train

q1 <- apply(d_test, 2, function(x) length(table(is.na(x))))
missing_variables_test <- names(d_test)[q1>1]
missing_variables_test

table(missing_variables_test %in% missing_variables_train)

d_train$country_code <- NULL
d_test$country_code <- NULL
d_train$desk_id <- NULL
d_test$desk_id <- NULL

d_train <- d_train[!is.na(d_train$bought),]

for(i in missing_variables_train)
{
  d_train[[i]][is.na(d_train[[i]])] <- paste0("Unk",i)
  d_test[[i]][is.na(d_test[[i]])] <- paste0("Unk",i)
  
}

date_variables <- c("start_date", "creation_date", "sell_date")

base_date <- ymd(20040401)

for(i in date_variables)
{
  d_train[[i]] <- ymd(d_train[[i]])
  d_test[[i]] <- ymd(d_test[[i]])
}



d_train$duration <- d_train$days_from_sell_date - d_train$days_from_start_date
d_test$duration <- d_test$days_from_sell_date - d_test$days_from_start_date

for(i in date_variables)
{
  d_train[[i]] <- NULL
  d_test[[i]] <- NULL
}



predCorr <- cor(d_train_full[,inputSet])
highCorr <- findCorrelation(predCorr, .99)
inputSet <- inputSet[-highCorr]

cat_var <- names(d_train)[c(2,3,6,9,10,11,12)]
for(i in cat_var)
{
  d_train[[i]] <- as.factor(d_train[[i]])
  d_test[[i]] <- as.factor(d_test[[i]])
}

setdiff(names(d_train), cat_var)

d_train <- as.data.frame(d_train)
d_test <- as.data.frame(d_test)
sp <- createDataPartition(d_train$return, p = 0.7)[[1]]

trainT <- d_train[sp,]
trainV <- d_train[-sp,]

set.seed(100)
indx <- createFolds(trainT$return, returnTrain = TRUE)
ctrl <- trainControl(method = "cv")

inputSetfil <- names(trainT)[c(2:12,14:18)]

lmTune0 <- train(x = trainT[,inputSetfil], y = trainT$return,
                 method = "lm",
                 trControl = ctrl)

lmTune0


##########################################################################3
#cat_variables <- names(d_train)[c(2,3,7,11,13,14,15)]

for(i in cat_variables)
{
  d_train[[i]] <- as.factor(d_train[[i]])
  d_test[[i]] <- as.factor(d_test[[i]])
}


date_variables <- c("start_date", "creation_date", "sell_date")

for(i in date_variables)
{
  d_train[[i]] <- ymd(d_train[[i]])
  d_test[[i]] <- ymd(d_test[[i]])
}

base_date <- ymd(20040401)

for(i in date_variables)
{
  nm <- paste0("days_from_",i)
  d_train[[nm]] <- as.integer(d_train[[i]] - base_date)
  d_test[[nm]] <- as.integer(d_test[[i]] - base_date)
}

d_train$duration <- d_train$days_from_sell_date - d_train$days_from_start_date
d_test$duration <- d_test$days_from_sell_date - d_test$days_from_start_date

for(i in date_variables)
{
  d_train[[i]] <- NULL
  d_test[[i]] <- NULL
}

d_train <- as.data.frame(d_train)
d_test <- as.data.frame(d_test)

dummy_function <- dummyVars(~., data = d_train[cat_variables])
d_train_cat <- as.data.frame(predict(dummy_function, d_train[cat_variables]))
d_test_cat <- as.data.frame(predict(dummy_function, d_test[cat_variables]))

d_train_full <- cbind(d_train[setdiff(names(d_train),cat_variables)], d_train_cat)
d_test_full <- cbind(d_test[setdiff(names(d_test),cat_variables)], d_test_cat)

fullSet <- names(d_train_full)[c(2:5,7:37)]
d_train_full[,fullSet] <- apply(d_train_full[,fullSet],2,as.numeric)
d_test_full[,fullSet] <- apply(d_test_full[,fullSet],2,as.numeric)
d_train_full$return <- as.numeric(d_train_full$return)

d_train_full <- d_train_full[!is.na(d_train_full$sold),]

inputSet <- names(d_train_full)[c(2,5,7:37)]
  
predCorr <- cor(d_train_full[,inputSet])
highCorr <- findCorrelation(predCorr, .99)
inputSet <- inputSet[-highCorr]

d_train_fil <- cbind(d_train_full[,c(1,3,4,6)],d_train_full[,inputSet])
d_test_fil <- cbind(d_test_full[,c(1,3,4)], d_test_full[,inputSet])

#####################


sp <- createDataPartition(d_train_fil$return, p = 0.7)[[1]]

trainT <- d_train_fil[sp,]
trainV <- d_train_fil[-sp,]

set.seed(100)
indx <- createFolds(trainT$return, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

inputSetfil <- names(trainT)[c(2,3,5:28)]

lmTune0 <- train(x = trainT[,inputSetfil], y = trainT$return,
                 method = "lm",
                 trControl = ctrl)

lmTune0

testResults <- data.frame(obs = trainV$return,
                          Linear_Regression = predict(lmTune0, trainV))

######################################PLS
## Run PLS and PCR on solubility data and compare results
set.seed(100)
plsTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                 method = "pls",
                 tuneGrid = expand.grid(ncomp = 1:20),
                 trControl = ctrl)

plsTune

testResults$PLS <- predict(plsTune, trainV)
#####################
set.seed(100)
pcrTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                 method = "pcr",
                 tuneGrid = expand.grid(ncomp = 1:35),
                 trControl = ctrl)
pcrTune     

#########################################penalized models

ridgeGrid <- expand.grid(lambda = seq(0, .1, length = 15))

set.seed(100)
ridgeTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                   method = "ridge",
                   tuneGrid = ridgeGrid,
                   trControl = ctrl,
                   preProc = c("center", "scale"))


########################################################################################################################
########################################################################################################################
##############################trees######################################################
library(rpart)

cartTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                  method = "rpart",
                  tuneLength = 100,
                  trControl = ctrl, metric = "Rsquared")
cartTune
## cartTune$finalModel


### Plot the tuning results
plot(cartTune, scales = list(x = list(log = 10)))

cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
cartImp

### Save the test set results in a data frame                 
testResults <- data.frame(obs = trainV$return,
                          CART = predict(cartTune, trainV))

### Tune the conditional inference tree

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.60, .99, length = 20))))

set.seed(100)
ctreeTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                   method = "ctree",
                   tuneGrid = cGrid,
                   trControl = ctrl, metric = "Rsquared")
ctreeTune
plot(ctreeTune)

##ctreeTune$finalModel               
plot(ctreeTune$finalModel)
testResults$cTree <- predict(ctreeTune, trainV)

### Section 8.2 Regression Model Trees and 8.3 Rule-Based Models

### Tune the model tree. Using method = "M5" actually tunes over the
### tree- and rule-based versions of the model. M = 10 is also passed
### in to make sure that there are larger terminal nodes for the
### regression models.
trainV$libor_rate[is.na(trainV$libor_rate)] <- 100000
trainT$libor_rate[is.na(trainT$libor_rate)] <- 100000

set.seed(100)
m5Tune <- train(x = trainT[,inputSetfil], y = trainT$return,
                method = "M5",
                trControl = ctrl, metric = "Rsquared",
                control = Weka_control(M = 100))
m5Tune

plot(m5Tune)
#################################################treebag
set.seed(100)

treebagTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                     method = "treebag",
                     nbagg = 150,
                     trControl = ctrl, metric = "Rsquared")

treebagTune

testResults$treebag <- predict(treebagTune, trainV)

############################################################
mtryGrid <- data.frame(mtry = floor(seq(10, ncol(trainT[,inputSetfil]), length = 10)))


### Tune the model using cross-validation
set.seed(100)
rfTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                method = "rf",
                tuneGrid = mtryGrid,
                ntree = 100,
                importance = TRUE,
                trControl = ctrl, metric = "Rsquared")
rfTune

plot(rfTune)

rfImp <- varImp(rfTune, scale = FALSE)
rfImp

testResults$rf <- predict(rfTune, trainV)

R2(testResults$obs,testResults$rf)

#####################################################################


### Tune the conditional inference forests
set.seed(100)
condrfTune <- train(x = trainT[,inputSetfil], y = trainT$return,
                    method = "cforest",
                    tuneGrid = mtryGrid,
                    controls = cforest_unbiased(ntree = 1000), metric = "Rsquared",
                    trControl = ctrl)
condrfTune

plot(condrfTune)

