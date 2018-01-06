library(caret)
library(data.table)
#library(pROC)
#the following line will create a local 4-node snow cluster
library(doParallel)
workers=makeCluster(detectCores(),type="SOCK")
registerDoParallel(workers)

foreach(i=1:4) %dopar% Sys.getpid()


setwd("C:/Users/vjha3/vikas/misc/hackerearth/brainwaves/problem1/")

d_train <- fread("./input/train.csv", na.strings = c("", " ", "NA"))
d_test <- fread("./input/test.csv", na.strings = c("", " ", "NA"))

#View(d_train)
apply(d_train, 2, FUN = function(x) length(unique(x)))
apply(d_test, 2, FUN = function(x) length(unique(x)))

cat_variables <- names(d_train)[c(3,4,7,8,9,10,13,15,16,17)]

missing_levels <- lapply(cat_variables, FUN = function(x){unique(d_test[[x]])[!unique(d_test[[x]]) %in% unique(d_train[[x]])]})
missing_levels

train_d <- d_train
test_d <- d_test

d_train <- d_train[d_train$pf_category != "E",]
d_train <- d_train[d_train$type != "G",]

d_train$country_code <- NULL
d_test$country_code <- NULL

cat_variables <- names(d_train)[c(3,4,7,8,9,12,14,15,16)]
apply(d_train[,cat_variables, with = F], 2, FUN = function(x) length(unique(x)))
apply(d_test[,cat_variables, with = F], 2, FUN = function(x) length(unique(x)))

lapply(cat_variables, FUN = function(x){table(d_test[[x]] %in% d_train[[x]])})
lapply(cat_variables, FUN = function(x){table(d_test[[x]])})

dt1 <- merge(d_test, d_train, by = c("office_id", "pf_category", "start_date", "euribor_rate", "currency", "libor_rate", "creation_date",
                                     "indicator_code", "sell_date","type", "hedge_value", "status" ))
dt2 <- merge(d_test, d_train, by = c("office_id", "pf_category", "start_date", "euribor_rate", "currency", "libor_rate", "creation_date",
                                     "indicator_code", "sell_date","type", "hedge_value"))
dt1.1 <- dt1[,c(13,21), with = F]
dt1.1 <- unique(dt1.1)

port_no <- data.frame(pno = unique(dt1.1$portfolio_id.x), return = NA)

for(i in 1:length(port_no$pno))
{
  x <- dt1.1[dt1.1$portfolio_id.x == port_no$pno[i],]
  port_no$return[i] <- median(x$return)
}
  
write.csv(port_no, "./output/direct_return_test.csv", row.names = F)
d_test <- d_test[!(d_test$portfolio_id %in% dt1.1$portfolio_id.x),]
################################################
pf_names <- c("A", "B")

for(i in 1:length(pf_names))
{
  X <- train_d[train_d$type == pf_names[i],]
  assign(pf_names[i],X)
}

AB <- merge(A, B, by = c("start_date", "euribor_rate", "currency", "libor_rate", "creation_date",
                         "indicator_code", "sell_date", "hedge_value", "status", "type" ))

