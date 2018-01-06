library(caret)
library(data.table)
#library(pROC)
#the following line will create a local 4-node snow cluster

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
# intrain <- createDataPartition(d_train$target, p = 0.75)[[1]]
# training <- d_train[intrain,]
# testing <- d_train[-intrain,]

fullSet <- names(d_train)[2:43]

#########################################################################

library(h2o)
localH2O <- h2o.init(nthreads = -1, min_mem_size = "2g")
h2o.init(min_mem_size = "2g")

#data to h2o cluster
train.h2o <- as.h2o(d_train)
test.h2o <- as.h2o(d_test)

x.indep = c(2:43)
y.dep = 44

gbm <- h2o.gbm(x = x.indep, y = y.dep, training_frame = train.h2o)

h2o.auc(h2o.performance(gbm, xval = TRUE))
h2o.performance(gbm)
predict.gbm <- as.data.frame(h2o.predict(gbm, test.h2o))
test_output <- data.frame(transaction_id = d_test$transaction_id, target = predict.gbm$X1)
write.csv(test_output, "./output/gbmh2o_base.csv", row.names = F, quote = F)


######################cv################
hyper_params = list( max_depth = c(2,4,6,8))


grid.gbm <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  ## full Cartesian hyper-parameter search
  search_criteria = list(strategy = "Cartesian"),
    ## which algorithm to run
  algorithm="gbm",
  nfolds = 5,
    ## identifier for the grid, to later retrieve it
  grid_id="depth_grid",
    ## standard model parameters
  x = x.indep, y = y.dep,
  training_frame = train.h2o, 
  ## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 5000,                                                            
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  ## sample 80% of rows per tree
  sample_rate = 0.8,                                                       
  ## sample 80% of columns per split
  col_sample_rate = 0.8, 
  ## fix a random number generator seed for reproducibility
  seed = 1234,                                                             
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5,
  stopping_tolerance = 1e-4,
  stopping_metric = "AUC", 
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10                                                
)

grid.gbm

sortedGrid <- h2o.getGrid("depth_grid", sort_by="auc", decreasing = TRUE)
sortedGrid

topDepths = sortedGrid@summary_table$max_depth[1:4]                       
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))


###############################

hyper_params = list( 
  ## restrict the search to the range of max_depth established above
  max_depth = 6,                    
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2,1,0.01),                                         
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2,1,0.01),                                         
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = seq(0.5,1,0.01),                                
  ## search a large space of how column sampling per split should change as a function of the depth of the split
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),                      
  ## search a large space of the number of min rows in a terminal node
  min_rows = 2^seq(0,log2(nrow(train.h2o))-1,1),                                 
  ## search a large space of the number of bins for split-finding for continuous and integer columns
  nbins = 2^seq(4,10,1),                                                     
  ## search a large space of the number of bins for split-finding for categorical columns
  nbins_cats = 2^seq(4,12,1),                                                
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = c(0,1e-8,1e-6,1e-4),                               
  ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)

search_criteria = list(
  ## Random grid search
  strategy = "RandomDiscrete",      
  ## limit the runtime to 60 minutes
  max_runtime_secs = 7200,         
  ## build no more than 100 models
  max_models = 100,                  
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1234,                        
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 5,                
  stopping_metric = "AUC",
  stopping_tolerance = 1e-3
)


gridgbm2 <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria,
  ## which algorithm to run
  algorithm = "gbm",
  ## identifier for the grid, to later retrieve it
  grid_id = "final_grid", 
  ## standard model parameters
  x = x.indep, y = y.dep, 
  training_frame = train.h2o, 
  #validation_frame = valid,
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 5000,                                                            
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
  max_runtime_secs = 7200,                                                 
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  nfolds = 5,
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                                             
)


## Sort the grid models by AUC
sortedGrid <- h2o.getGrid("final_grid", sort_by = "auc", decreasing = TRUE)    
sortedGrid

for (i in 1:5) {
  gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
  print(h2o.auc(h2o.performance(gbm, valid = TRUE)))
}

gbm <- h2o.getModel(sortedGrid@model_ids[[1]])


predict.gbm <- as.data.frame(h2o.predict(gbm, test.h2o))
test_output <- data.frame(transaction_id = d_test$transaction_id, target = predict.gbm$X1)
head(test_output)
write.csv(test_output, "./output/gbmh2o_base_cv_updated.csv", row.names = F, quote = F)
