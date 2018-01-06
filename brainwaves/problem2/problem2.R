library(caret)
library(data.table)
library(pROC)
#the following line will create a local 4-node snow cluster
library(doParallel)
workers=makeCluster(detectCores(),type="SOCK")
registerDoParallel(workers)

foreach(i=1:4) %dopar% Sys.getpid()


setwd("C:/Users/vjha3/vikas/misc/hackerearth/brainwaves/problem2/")

d_train <- fread("./input/train.csv", na.strings = c("", " ", "NA"))
d_test <- fread("./input/test.csv", na.strings = c("", " ", "NA"))

d_train <- as.data.frame(d_train)
d_test <- as.data.frame(d_test)
## A function to find and remove zero-variance ("ZV") predictors
noZV <- function(x) {
  keepers <- unlist(lapply(x, function(x) length(unique(x)) > 1))
  x[,names(keepers)[keepers],drop = FALSE]
}

#View(d_train)
apply(d_train, 2, FUN = function(x) length(unique(x)))

d_train <- noZV(d_train)
d_test <- d_test[,names(d_test) %in% names(d_train)]

cat_variables <- names(d_train)[9:43]

##listing the variables with missing values
q <- apply(d_train, 2, function(x) length(table(is.na(x))))
missing_variables <- names(d_train)[q>1]

q1 <- apply(d_test, 2, function(x) length(table(is.na(x))))
missing_variables1 <- names(d_test)[q1>1]


###missing values will be converted to "Unk"
d_train$cat_var_1[is.na(d_train$cat_var_1)] <- "Unk_cat_var_1"
d_train$cat_var_3[is.na(d_train$cat_var_3)] <- "Unk_cat_var_3"
d_train$cat_var_8[is.na(d_train$cat_var_8)] <- "Unk_cat_var_8"
xx <- sample(1:nrow(d_train),1)
d_train[xx,cat_variables] <- paste0("Unk_cat_var_", seq(1,35))

d_test$cat_var_1[is.na(d_test$cat_var_1)] <- "Unk_cat_var_1"
d_test$cat_var_3[is.na(d_test$cat_var_3)] <- "Unk_cat_var_3"
d_test$cat_var_8[is.na(d_test$cat_var_8)] <- "Unk_cat_var_8"
d_test$cat_var_6[is.na(d_test$cat_var_6)] <- "zs"
d_test$cat_var_1[d_test$cat_var_1 %in% c("gz", "jk", "yw")] <- "Unk_cat_var_1"

#missing_levels <- lapply(cat_variables, FUN = function(x){levels(d_test[[x]])[!levels(d_test[[x]]) %in% levels(d_train[[x]])]})
missing_levels <- lapply(cat_variables, FUN = function(x){unique(d_test[[x]])[!unique(d_test[[x]]) %in% unique(d_train[[x]])]})
#df_missing_levels <- data.frame(variable  = cat_variables, missing_levels = 0)
#df_missing_levels <- lapply(1:nrow(df_missing_levels), FUN = function(x){df_missing_levels$missing_levels[x] <- missing_levels[[x]]})
#missing_prop <- list()

for(i in 1:length(missing_levels))
{
  q <- missing_levels[[i]]
  if(length(q)>0)
  {
    for(j in 1:length(q))
    {
      print(paste(cat_variables[i],q[j],sep = ":" ))
      print(prop.table(table(d_test[cat_variables[i]] == q[j])))
      d_test[cat_variables[i]][d_test[cat_variables[i]] == q[j]] <- paste0("Unk_cat_var_",i)
    }
  }
}


for(i in cat_variables)
{d_train[[i]] <- factor(d_train[[i]])
d_test[[i]] <- factor(d_test[[i]])}

d_train$target <- factor(d_train$target)
levels(d_train$target) <- c("X0", "X1")
#dummiez1 <- dummyVars(~., data = d_train[cat_variables[1]], levelsOnly = T)
#d_train_dummy1 <- predict(dummiez1, newdata = d_train)

set.seed(123)
intrain <- createDataPartition(d_train$target, p = 0.75)[[1]]
training <- d_train[intrain,]
testing <- d_train[-intrain,]

fullSet <- names(d_train)[2:43]
######################################################################################
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
set.seed(476)
rpartFit <- train(x = d_train[,fullSet], 
                  y = d_train$target,
                  method = "rpart",
                  tuneLength = 30,
                  metric = "ROC",
                  trControl = ctrl)

rpartFit

library(partykit)
plot(as.party(rpartFit$finalModel))

rpartCM <- confusionMatrix(rpartFit, norm = "none")
rpartCM
rpartRoc <- roc(response = rpartFit$pred$obs,
                predictor = rpartFit$pred$X1,
                levels = rev(levels(rpartFit$pred$obs)))
rpart_test_pred <- predict(rpartFit$finalModel, newdata = testing, type = "class")
rpart_test_pred_df <- data.frame(obs = testing$target, pred = rpart_test_pred)
rpart_test_pred_df$prob <- predict(rpartFit$finalModel, newdata = testing, type = "prob")[,2]
confusionMatrix(rpart_test_pred_df$pred, rpart_test_pred_df$obs)
rpartROC_test <- roc(response = rpart_test_pred_df$obs, predictor = rpart_test_pred_df$prob, levels = rev(levels(rpart_test_pred_df$obs)))
head(rpart_test_pred_df)

rpart_d_test_pred <- predict(rpartFit$finalModel, newdata = d_test, type = "prob")
test_output <- data.frame(transaction_id = d_test$transaction_id, target = rpart_d_test_pred[,2])
write.csv(test_output, "./output/rpart30LGOCV.csv", row.names = F, quote = F)

######################################

set.seed(476)
j48Fit <- train(x = d_train[,fullSet], 
                y = d_train$target,
                method = "J48",
                metric = "ROC",
                trControl = ctrl)


#######################################

set.seed(476)
partFit <- train(x = d_train[,fullSet], 
                 y = d_train$target,
                 method = "PART",
                 metric = "ROC",
                 trControl = ctrl)


#################################
# 
# mtryValues <- c(5, 10, 20, 32, 50, 100, 250, 500, 1000)
# set.seed(476)
# rfFit <- train(x = d_train[,fullSet], 
#                y = d_train$target,
#                method = "rf",
#                ntree = 500,
#                tuneGrid = data.frame(mtry = mtryValues),
#                importance = TRUE,
#                metric = "ROC",
#                trControl = ctrl)
# rfFit

#cant use rf as factor levels are more than it can handle
#########################################

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       #n.trees = (1:20)*100,
                       n.trees = 100,
                       shrinkage = c(.01, .1), n.minobsinnode = 1000)

gbmGrid <- expand.grid(interaction.depth = c(1, 3),
                       #n.trees = (1:20)*100,
                       n.trees = 100,
                       shrinkage = c(.01, .1), n.minobsinnode = 10000)
set.seed(476)
gbmFit <- train(x = d_train[,fullSet], 
                y = d_train$target,
                method = "gbm",
                tuneGrid = gbmGrid,
                metric = "ROC",
                verbose = F,
                trControl = ctrl)
gbm_d_test_pred <- predict(gbmFit$finalModel, newdata = d_test, type = "prob")
test_output <- data.frame(transaction_id = d_test$transaction_id, target = rpart_d_test_pred[,2])
write.csv(test_output, "./output/rpart30LGOCV.csv", row.names = F, quote = F)

#################################################
c50Grid <- expand.grid(trials = c(1:9, (1:10)*10),
                       model = c("tree", "rules"),
                       winnow = c(TRUE, FALSE))
set.seed(476)
c50Fit <- train(d_train[,fullSet], d_train$target,
                method = "C5.0",
                tuneGrid = c50Grid,
                metric = "ROC",
                verbose = FALSE,
                trControl = ctrl)


####################################################

set.seed(476)
treebagFit <- train(x = d_train[,fullSet], 
                    y = d_train$target,
                    method = "treebag",
                    nbagg = 50,
                    metric = "ROC",
                    trControl = ctrl)
treebagFit

########################################################
library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()

#data to h2o cluster
train.h2o <- as.h2o(d_train)
test.h2o <- as.h2o(d_test)

x.indep = c(2:43)
y.dep = 44

#GBM
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 10000, max_depth = 4, learn_rate = 0.005, seed = 1122,
                       stopping_metric = "AUC")
)
h2o.performance (gbm.model)

predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))

test_output <- data.frame(transaction_id = d_test$transaction_id, target = predict.gbm$X1)
write.csv(test_output, "./output/gbmh2o.csv", row.names = F, quote = F)

####################################################################
################tune gbm##########################################
