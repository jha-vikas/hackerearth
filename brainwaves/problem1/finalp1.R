library(caret)
library(data.table)
#library(pROC)
#the following line will create a local 4-node snow cluster
library(doParallel)
library(lubridate)

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

base_date <- ymd(20040401)

d_train$sell_date <- ymd(d_train$sell_date)
d_train$year <- year(d_train$sell_date)
d_train$month <- month(d_train$sell_date)
d_train$type <- as.factor(d_train$type)
d_train$currency <- as.factor(d_train$currency)
d_train$dayssince <- as.integer(d_train$sell_date - base_date)
d_train$invday <- (d_train$dayssince)^-1

d_test$sell_date <- ymd(d_test$sell_date)
d_test$year <- year(d_test$sell_date)
d_test$month <- month(d_test$sell_date)
d_test$type <- as.factor(d_test$type)
d_test$currency <- as.factor(d_test$currency)
d_test$dayssince <- as.integer(d_test$sell_date - base_date)
d_test$invday <- (d_test$dayssince)^-1

libor_imp <- train(x = d_train[!is.na(d_train$libor_rate),c(6,7,17:20)], y = d_train$libor_rate[!is.na(d_train$libor_rate)], method = "lm", trControl = trainControl(method = "cv"))
summary(libor_imp)

plot(d_train$libor_rate[!is.na(d_train$libor_rate)], libor_imp$finalModel$fitted.values)
pred_test_test <- predict(libor_imp$finalModel, newdata = d_test[!is.na(d_test$libor_rate),])

RMSE(pred_test_test, d_test$libor_rate[!is.na(d_test$libor_rate)])
R2(pred_test_test, d_test$libor_rate[!is.na(d_test$libor_rate)])


pred_lib <- predict(libor_imp$finalModel, newdata = d_train[is.na(d_train$libor_rate),])

d_train$libor_rate[is.na(d_train$libor_rate)] <- pred_lib
d_test$libor_rate[is.na(d_test$libor_rate)] <- predict(libor_imp$finalModel, newdata = d_test[is.na(d_test$libor_rate),])

d_train %>% summarize_all(funs(sum(is.na(.)) / length(.)))
d_test %>% summarize_all(funs(sum(is.na(.)) / length(.)))

names(d_train)

d_train[,c(2,3,7,11,13,14,15,17,18)] %>% apply(MARGIN = 2, table)
d_test[,c(2,3,7,11,13,14,15,16,17)] %>% apply(MARGIN = 2, table)

d_train <- d_train[d_train$pf_category != "E",]
d_train <- d_train[d_train$type != "G",]


d_train <- d_train[!is.na(d_train$bought),]

q <- apply(d_train, 2, function(x) length(table(is.na(x))))
missing_variables_train <- names(d_train)[q>1]
missing_variables_train

q1 <- apply(d_test, 2, function(x) length(table(is.na(x))))
missing_variables_test <- names(d_test)[q1>1]
missing_variables_test

table(missing_variables_test %in% missing_variables_train)

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

d_train$days_from_start_date <- as.integer(d_train$start_date - base_date)
d_test$days_from_start_date <- as.integer(d_test$start_date - base_date)

d_train$duration <- as.integer(d_train$dayssince - d_train$days_from_start_date)
d_test$duration <- as.integer(d_test$dayssince - d_test$days_from_start_date)

for(i in date_variables)
{
  d_train[[i]] <- NULL
  d_test[[i]] <- NULL
}

cat_var <- names(d_train)[c(2,3,6,9,10,11,12)]
for(i in cat_var)
{
  d_train[[i]] <- as.factor(d_train[[i]])
  d_test[[i]] <- as.factor(d_test[[i]])
}


#d_train$price_ratio <- d_train$sold/d_train$bought
d_train$basic_return <- ((d_train$sold/d_train$bought) - 1)
d_test$basic_return <- ((d_test$sold/d_test$bought) - 1)

cont_var <- setdiff(setdiff(names(d_train), cat_var), c("portfolio_id", "return"))

d_train <- as.data.frame(d_train)
d_test <- as.data.frame(d_test)

predCorr <- cor(d_train[cont_var])
highCorr <- findCorrelation(predCorr, .7)
cont_var <- cont_var[-highCorr]

finalSet <- c(cont_var, cat_var)

direct_return <- fread("file:///C:/Users/vjha3/vikas/misc/hackerearth/brainwaves/problem1/output/direct_return_test.csv")
names(direct_return)[1] <- "portfolio_id" 
d_test <- d_test[!(d_test$portfolio_id %in% direct_return$portfolio_id),]
#################################################################################



####################################################################################3

## Run PLS and PCR on solubility data and compare results
set.seed(sample(1:1000,1))

sp <- createDataPartition(d_train$return, p = 0.8)[[1]]

trainT <- d_train[sp,]
trainV <- d_train[-sp,]
set.seed(sample(1000:2000,1))
plsTune <- train(x = trainT[,finalSet], y = trainT$return,
                 method = "pcr",
                 tuneGrid = expand.grid(ncomp = 1:20),
                 trControl = ctrl, metric = "Rsquared")
plsTune

R2(trainV$return, predict(plsTune, trainV))

plot(trainV$return, predict(plsTune, trainV))

head(data.frame(obs = trainV$return, pred = predict(plsTune, trainV)),20)

#####################

##################################################################################3#

d_test_or <- fread("./input/test.csv", na.strings = c("", " ", "NA"))
test_resut <- data.frame(portfolio_id = d_test_or$portfolio_id, stringsAsFactors = F)
test_resut <- left_join(test_resut, df_test_pred, "portfolio_id")

write.csv(test_resut, "./output/lm_direct_mix_v4.csv", row.names = F, quote = F)
