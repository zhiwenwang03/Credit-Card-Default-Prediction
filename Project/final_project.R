data_raw <- read.table("C:/Users/pbcs/Desktop/Slides-STAT 4052/Project/credit_default32.txt",header=T)
factor_var <- c("SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","Default")
data_raw[factor_var] <- lapply(data_raw[factor_var],factor)
data1 <- data_raw
data1 <- subset(data1,select=-ID)
num_var <- names(data1)[sapply(data1,is.numeric)]

for (a in num_var){
  data1[[a]][is.na(data1[[a]])] <- mean(data_raw[[a]],na.rm=T)
}
data1 <- data1[complete.cases(data1),]

# Construct a suitable logistic regression model to predict Default.
set.seed(123)

train_idx <- sample(1:nrow(data1),round(0.7*nrow(data1)))
train <- data1[train_idx,]
test <- data1[-train_idx,]

logit_full <- glm(Default ~ .,data=train,family="binomial")
summary(logit_full)

# Backward stepwise
model_back <- step(logit_full,direction="backward")
summary(model_back)

# Forward stepwise (Failure)
logit_null <- glm(Default ~ 1,data=train,family="binomial")
step(logit_null,scope=list(lower=~1,upper=~.),direction="forward")

# LASSO logistic regression
library(glmnet)
x_train <- model.matrix(Default ~ .,train)[,-1]
y <- as.numeric(train$Default)-1

set.seed(123)
cvfit <- cv.glmnet(x_train,y,family="binomial",alpha=1)

best_lambda <- cvfit$lambda.min
model_lasso <- glmnet(x_train,y,family="binomial",lambda=best_lambda)
coef(model_lasso)

# Compare different models
## Backward model
pred_back_prob <- predict(model_back,newdata=test,type="response")

# Convert to class
pred_back_class <- ifelse(pred_back_prob>0.5,1,0)

# Truth
truth <- as.numeric(test$Default)-1

acc_back <- mean(pred_back_class == truth)
acc_back

library(ROCR)
pred_back <- prediction(pred_back_prob,truth)
perf_back <- performance(pred_back,"tpr","fpr")

plot(perf_back,colorize=T,main="Backward Stepwise")

AUC_back <- performance(pred_back,"auc")@y.values[[1]]
AUC_back

## LASSO model
x_test <- model.matrix(Default ~ .,test)[,-1]
pred_lasso_prob <- predict(model_lasso,x_test,type="response")

pred_lasso_class <- ifelse(pred_lasso_prob>0.5,1,0)
acc_lasso <- mean(pred_lasso_class == truth)
acc_lasso

pred_lasso <- prediction(pred_lasso_prob,truth)
perf_lasso <- performance(pred_lasso,"tpr","fpr")

plot(perf_lasso,colorize=T,main="LASSO")

AUC_lasso <- performance(pred_lasso,"auc")@y.values[[1]]
AUC_lasso

# KNN
library(class)
library(caret)
library(fastDummies)
set.seed(123)
train_knn <- train
test_knn <- test

train_knn$Default <- as.numeric(train_knn$Default) - 1
test_knn$Default <- as.numeric(test_knn$Default) - 1

# Create dummy variables
cat_vars <- names(train_knn)[sapply(train_knn,is.factor)]
train_dummy <- dummy_cols(train_knn,remove_selected_columns = TRUE,remove_first_dummy = TRUE)
test_dummy <- dummy_cols(test_knn,remove_selected_columns = TRUE,remove_first_dummy = TRUE)

train_y <- factor(train_dummy$Default,levels = c(0,1))
test_y <- factor(test_dummy$Default,levels = c(0,1))

train_dummy$Default <- NULL
test_dummy$Default  <- NULL

num_vars <- names(train_knn)[sapply(train_knn,is.numeric) & names(train_knn) != "Default"]

preProcValues <- preProcess(train_knn[,num_vars],method = c("center", "scale"))
train_dummy_scaled <- train_dummy
test_dummy_scaled <- test_dummy

train_dummy_scaled[,num_vars] <- predict(preProcValues,train_knn[,num_vars])
test_dummy_scaled[,num_vars] <- predict(preProcValues,test_knn[,num_vars])

k_values <- seq(1,50,by=2)
acc_list <- c()

for (k in k_values){
  pred_k <- knn(train_dummy_scaled,test_dummy_scaled,cl = train_y,k = k)
  acc_list <- c(acc_list,mean(pred_k == test_y))
}

best_k <- k_values[which.max(acc_list)]
best_k

pred_knn <- knn(train_dummy_scaled,test_dummy_scaled,cl = train_y,k = best_k)

# Accuracy
acc_knn <- mean(pred_knn == test_y)
acc_knn

knn_prob <- attributes(knn(train_dummy_scaled, test_dummy_scaled,
                           cl = train_y, k = best_k, prob = TRUE))$prob
knn_prob <- ifelse(pred_knn == "1", knn_prob, 1 - knn_prob)
knn_prob <- as.numeric(knn_prob)

truth_knn <- as.numeric(test_y) - 1
pred_knn_ <- prediction(knn_prob,truth_knn)
AUC_knn <- performance(pred_knn_,"auc")@y.values[[1]]
AUC_knn

# ROC curve
perf_knn <- performance(pred_knn_,"tpr","fpr")
plot(perf_knn,colorize = TRUE,main = "KNN")

# Random forest
library(randomForest)
set.seed(111)
m1 <- randomForest(Default~.,data=train,mtry=floor(sqrt(ncol(train)-1)),importance=T)
m1

## Predict on test set
pred_rf <- predict(m1,newdata=test,type="class")
mean(pred_rf!=test$Default)

varImpPlot(m1)

pred_rf_prob <- predict(m1,newdata=test,type="prob")[,2]
pred_rf <- prediction(pred_rf_prob,truth)
perf_rf <- performance(pred_rf,"tpr","fpr")

plot(perf_rf,colorize=T,main="Random Forest")

AUC_rf <- performance(pred_rf,"auc")@y.values[[1]]
AUC_rf

# Bagging
set.seed(222)
m2 <- randomForest(Default ~ .,data=train,mtry=ncol(train)-1,importance=T)
m2

## Predict on test set
pred_bg <- predict(m2,newdata=test,type="class")
mean(pred_bg!=test$Default)

varImpPlot(m2)

pred_bg_prob <- predict(m2,newdata=test,type="prob")[,2]
pred_bg <- prediction(pred_bg_prob,truth)
perf_bg <- performance(pred_bg,"tpr","fpr")

plot(perf_bg,colorize=T,main="Bagging")

AUC_bg <- performance(pred_bg,"auc")@y.values[[1]]
AUC_bg

# Boosting
library(gbm)
set.seed(333)
train_bt <- train
test_bt <- test
train_bt$Default <- as.numeric(train_bt$Default)-1
test_bt$Default <- as.numeric(test_bt$Default)-1

m3 <- gbm(Default ~ .,data=train_bt,distribution="bernoulli",n.trees=5000,interaction.depth=3,shrinkage=0.1)
summary(m3)

## Predict on test set
pred_bt <- predict(m3,test_bt,type="response",n.trees=5000)
pred_bt_class <- ifelse(pred_bt>0.5,1,0)
pred_bt_class <- factor(pred_bt_class,levels=c(0,1))
test_bt$Default <- factor(test_bt$Default,levels=c(0,1))
mean(pred_bt_class!=test_bt$Default)

pred_bt_ <- prediction(pred_bt,truth)
perf_bt <- performance(pred_bt_,"tpr","fpr")

plot(perf_bt,colorize=T,main="Boosting")

AUC_bt <- performance(pred_bt_,"auc")@y.values[[1]]
AUC_bt

# Iterative regression
data_iter <- data_raw
data_iter <- subset(data_iter,select=-ID)

# First impute variables
var_num <- names(data_iter)[sapply(data_iter,is.numeric)]
var_cat <- names(data_iter)[sapply(data_iter,is.factor)]

for (v in var_num){
  data_iter[[v]][is.na(data_iter[[v]])] <- mean(data_iter[[v]],na.rm=T)
}

f1 <- function(x)names(which.max(table(x)))
for (v in var_cat){
  data_iter[[v]][is.na(data_iter[[v]])] <- f1(data_iter[[v]])
}

library(nnet)
n_iter <- 10

for (i in 1:n_iter){
  for (v in var_num){
    missing_index <- is.na(data_iter[[v]])
    if(any(missing_index)){
      f2 <- as.formula(paste(v,"~."))
      model_num <- lm(f2,data=data_iter,subset=!missing_index)
      pred <- predict(model_num,data_iter[missing_index,])
      data_iter[[v]][missing_index] <- pred
    }
  }
  
  for(v in var_cat){
    missing_index <- is.na(data_iter[[v]])
    if(!any(missing_index)) next
    levels_count <- length(levels(data_iter[[v]]))
    
    # 2-level
    if(levels_count == 2){
      f3 <- as.formula(paste(v,"~."))
      model_bin <- glm(f3,data=data_iter,
                       family=binomial,
                       subset=!missing_index)
      p <- predict(model_bin,data_iter[missing_index,],type="response")
      pred_class <- ifelse(p > 0.5,
                           levels(data_iter[[v]])[2],
                           levels(data_iter[[v]])[1])
      data_iter[[v]][missing_index] <- pred_class
    }
    
    # multinomial
    else {
      f4 <- as.formula(paste(v,"~."))
      model_mul <- multinom(f4,data=data_iter,subset=!missing_index,trace=FALSE)
      pred_class <- predict(model_mul,data_iter[missing_index,])
      data_iter[[v]][missing_index] <- pred_class
    }
  }
}

summary(data_iter)
sum(is.na(data_iter))

# Repeat the analyses
set.seed(1234)

train_idx1 <- sample(1:nrow(data_iter),round(0.7*nrow(data_iter)))
train1 <- data_iter[train_idx1,]
test1 <- data_iter[-train_idx1,]

logit_full1 <- glm(Default ~ .,data=train1,family="binomial")
summary(logit_full1)

# Backward stepwise
model_back1 <- step(logit_full1,direction="backward")
summary(model_back1)

pred_back_prob1 <- predict(model_back1,newdata=test1,type="response")

# Convert to class
pred_back_class1 <- ifelse(pred_back_prob1>0.5,1,0)

# Truth
truth1 <- as.numeric(test1$Default)-1

acc_back1 <- mean(pred_back_class1 == truth1)
acc_back1

library(ROCR)
pred_back1 <- prediction(pred_back_prob1,truth1)
AUC_back1 <- performance(pred_back1,"auc")@y.values[[1]]
AUC_back1

# Forward stepwise (Failure)
logit_null1 <- glm(Default ~ 1,data=train1,family="binomial")
step(logit_null1,scope=list(lower=~1,upper=~.),direction="forward")

# LASSO logistic regression
library(glmnet)
x_train1 <- model.matrix(Default ~ .,train1)[,-1]
y1 <- as.numeric(train1$Default)-1

set.seed(123)
cvfit1 <- cv.glmnet(x_train1,y1,family="binomial",alpha=1)

best_lambda1 <- cvfit1$lambda.min
model_lasso1 <- glmnet(x_train1,y1,family="binomial",lambda=best_lambda1)

x_test1 <- model.matrix(Default ~ .,test1)[,-1]
pred_lasso_prob1 <- predict(model_lasso1,x_test1,type="response")

pred_lasso_class1 <- ifelse(pred_lasso_prob1>0.5,1,0)
acc_lasso1 <- mean(pred_lasso_class1 == truth1)
acc_lasso1

pred_lasso1 <- prediction(pred_lasso_prob1,truth1)
perf_lasso1 <- performance(pred_lasso1,"tpr","fpr")

plot(perf_lasso1,colorize=T,main="LASSO")

AUC_lasso1 <- performance(pred_lasso1,"auc")@y.values[[1]]
AUC_lasso1


# KNN
library(class)
library(caret)
library(fastDummies)
set.seed(321)
train_knn1 <- train1
test_knn1 <- test1

train_knn1$Default <- as.numeric(train_knn1$Default) - 1
test_knn1$Default <- as.numeric(test_knn1$Default) - 1

# Create dummy variables
cat_vars1 <- names(train_knn1)[sapply(train_knn1,is.factor)]
train_dummy1 <- dummy_cols(train_knn1,remove_selected_columns = TRUE,remove_first_dummy = TRUE)
test_dummy1 <- dummy_cols(test_knn1,remove_selected_columns = TRUE,remove_first_dummy = TRUE)

train_y1 <- factor(train_dummy1$Default,levels = c(0,1))
test_y1 <- factor(test_dummy1$Default,levels = c(0,1))

train_dummy1$Default <- NULL
test_dummy1$Default  <- NULL

num_vars1 <- names(train_knn1)[sapply(train_knn1,is.numeric) & names(train_knn1) != "Default"]

preProcValues1 <- preProcess(train_knn1[,num_vars1],method = c("center", "scale"))
train_dummy_scaled1 <- train_dummy1
test_dummy_scaled1 <- test_dummy1

train_dummy_scaled1[,num_vars1] <- predict(preProcValues1,train_knn1[,num_vars1])
test_dummy_scaled1[,num_vars1] <- predict(preProcValues1,test_knn1[,num_vars1])

k_values1 <- seq(1,50,by=2)
acc_list1 <- c()

for (k in k_values1){
  pred_k1 <- knn(train_dummy_scaled1,test_dummy_scaled1,cl = train_y1,k = k)
  acc_list1 <- c(acc_list1,mean(pred_k1 == test_y1))
}

best_k1 <- k_values1[which.max(acc_list1)]
best_k1

pred_knn1 <- knn(train_dummy_scaled1,test_dummy_scaled1,cl = train_y1,k = best_k1)

# Accuracy
acc_knn1 <- mean(pred_knn1 == test_y1)
acc_knn1

knn_prob1 <- attributes(knn(train_dummy_scaled1, test_dummy_scaled1,
                           cl = train_y1, k = best_k1, prob = TRUE))$prob
knn_prob1 <- ifelse(pred_knn1 == "1", knn_prob1, 1 - knn_prob1)
knn_prob1 <- as.numeric(knn_prob1)

truth_knn1 <- as.numeric(test_y1) - 1
pred_knn_1 <- prediction(knn_prob1,truth_knn1)
AUC_knn1 <- performance(pred_knn_1,"auc")@y.values[[1]]
AUC_knn1

# ROC curve
perf_knn1 <- performance(pred_knn_1,"tpr","fpr")
plot(perf_knn1,colorize = TRUE,main = "KNN")


# Random forest
library(randomForest)
set.seed(1111)
m11 <- randomForest(Default~.,data=train1,mtry=floor(sqrt(ncol(train1)-1)),importance=T)
m11

## Predict on test set
pred_rf1 <- predict(m11,newdata=test1,type="class")
mean(pred_rf1!=test1$Default)

varImpPlot(m11)

pred_rf_prob1 <- predict(m11,newdata=test1,type="prob")[,2]
pred_rf1 <- prediction(pred_rf_prob1,truth1)
perf_rf1 <- performance(pred_rf1,"tpr","fpr")

plot(perf_rf1,colorize=T,main="Random Forest")

AUC_rf1 <- performance(pred_rf1,"auc")@y.values[[1]]
AUC_rf1

# Bagging
set.seed(2222)
m21 <- randomForest(Default ~ .,data=train1,mtry=ncol(train1)-1,importance=T)
m21

## Predict on test set
pred_bg1 <- predict(m21,newdata=test1,type="class")
mean(pred_bg1!=test1$Default)

varImpPlot(m21)

pred_bg_prob1 <- predict(m21,newdata=test1,type="prob")[,2]
pred_bg1 <- prediction(pred_bg_prob1,truth1)
perf_bg1 <- performance(pred_bg1,"tpr","fpr")

plot(perf_bg1,colorize=T,main="Bagging")

AUC_bg1 <- performance(pred_bg1,"auc")@y.values[[1]]
AUC_bg1

# Boosting
library(gbm)
set.seed(3333)
train_bt1 <- train1
test_bt1 <- test1
train_bt1$Default <- as.numeric(train_bt1$Default)-1
test_bt1$Default <- as.numeric(test_bt1$Default)-1

m31 <- gbm(Default ~ .,data=train_bt1,distribution="bernoulli",n.trees=5000,interaction.depth=3,shrinkage=0.1)
m31
summary(m31)

## Predict on test set
pred_bt1 <- predict(m31,test_bt1,type="response",n.trees=5000)
pred_bt_class1 <- ifelse(pred_bt1>0.5,1,0)
pred_bt_class1 <- factor(pred_bt_class1,levels=c(0,1))
test_bt1$Default <- factor(test_bt1$Default,levels=c(0,1))
mean(pred_bt_class1!=test_bt1$Default)

pred_bt_1 <- prediction(pred_bt1,truth1)
perf_bt1 <- performance(pred_bt_1,"tpr","fpr")

plot(perf_bt1,colorize=T,main="Boosting")

AUC_bt1 <- performance(pred_bt_1,"auc")@y.values[[1]]
AUC_bt1