ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
x.indep = c(2:43)
y.dep = 44

set.seed(476)
lrFit <- train(x = d_train[,x.indep], 
               y = d_train$target,
               method = "glm",
               metric = "ROC",
               trControl = ctrl)

