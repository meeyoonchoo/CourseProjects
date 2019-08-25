library(ggplot2)
library(caret)
library(dplyr)
library(MASS)
library("tree")
setwd("C:/Users/meeyoonchoo/Documents/BayPath/ADS635 DataMining1")
heart<- read.csv("C:/Users/meeyoonchoo/Documents/BayPath/ADS635 DataMining1/heart.csv", header=TRUE, sep=",")
dim(heart) #303, 14
head(heart)
str(heart)
names(heart)[1] <- "age"
summary(heart)
attach(heart)
hist(target, col=8)
table(sex,target)
barplot(table(sex,target),beside=TRUE)
ggplot(data=heart,aes(cp)) + geom_histogram()
boxplot(trestbps)
hist(trestbps,col=8)
hist(chol, col=8)
boxplot(chol)
ggplot(data=heart,aes(trestbps)) + geom_histogram()
ggplot(data=heart,aes(chol)) + geom_histogram()
pairs(~age+sex+cp+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope+ca+thal+target,heart)
cor(heart)

#classification :use logistic regression
#data division into train and test set
set.seed(123)
train <- sample(1:nrow(heart), .8*nrow(heart))
heart_train <- heart[train,]
heart_test <- heart[-train,]

#########################################
# Fitting a full Logistic Regression
#########################################
full.mod <- glm(target~., data = heart_train, family = "binomial")
summary(full.mod)

prob <- full.mod %>% predict(heart_test,type ="response")
predicted.class1 <- ifelse(prob>0.5,1,0)
mean(predicted.class1==heart_test$target) #accuracy=0.9180



# KNN Model
#k=10
require(class)
set.seed(1234)
train.x <-cbind(heart_train[,-14])
test.x <-cbind(heart_test[,-14])
train.y <-cbind(heart_train$target)

pred.knn <- knn(train.x,test.x,train.y,k=10)
table(pred.knn,heart_test$target)
mean(pred.knn==heart_test$target) #Accuracy=0.72131

#LDA
lda.fit <- lda(target~., data=heart_train)
lda.fit
summary(lda.fit)

y_true.test <-as.numeric(heart_test$target)

lda.pred <- predict(lda.fit,heart_test)
names(lda.pred) #"class" "posterior" "x"        
lda.pred$class
table(lda.pred$class,y_true.test)
mean(lda.pred$class==y_true.test) #Accuracy= 0.871

#QDA
qda.fit <-qda(target~., data=heart_train)
qda.fit
summary(qda.fit)

qda.pred <- predict(qda.fit,heart_test)
names(qda.pred) #"class" "posterior" "x"        
qda.pred$class
table(qda.pred$class,y_true.test)
mean(qda.pred$class==y_true.test) #Accuracy= 0.8387

# PCA
pr.out=prcomp(heart, scale=TRUE)
names(pr.out)
dim(pr.out$x)
biplot(pr.out,scale=0)
#compute standard deviation of each principal component
std_dev <- pr.out$sdev
#compute variance
pr_var <- std_dev^2
#check variance of 6 components
pr_var
#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex
#scree plot
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

heart2 <- cbind(heart, pr.out$x)
ggplot(heart2, aes(PC1, PC2, col = target, fill = target)) +
  stat_ellipse(geom = "polygon", col = "black", alpha = 0.5) +
  geom_point(shape = 21, col = "black")


#which component has the largest effect? 
loading_scores <- pr.out$rotation[,1]
note_scores <- abs(loading_scores)
note_scores_ranked <- sort(note_scores, decreasing = TRUE)
component_ranked <- names(note_scores_ranked)
component_ranked
pr.out$rotation[component_ranked,1] #diagonal has the largest effect

#Random Forest Model
library(randomForest)

# set the seed, and put aside a test set
set.seed(12345)
test_indis <- sample(1:nrow(heart), .20*nrow(heart))
test <- heart[test_indis, ]
training <- heart[-test_indis, ]
y_true <- as.numeric(test$target) # 0 No, 1 Yes

rf.fit <- randomForest(target~., data = training, n.tree = 10000)

x11()
varImpPlot(rf.fit)

importance(rf.fit)

y_hat <- predict(rf.fit, newdata = test, type = "response", distribution="binomial")
rf.pred <-ifelse(y_hat>0.5,1,0)
mean((y_true-rf.pred)^2) ## MSE=0.0667 
misclass_rf <- sum(abs(y_true- rf.pred))/length(rf.pred)
misclass_rf # 0.0667
mean(rf.pred==y_true) #Accuracy=0.9333

#SVM Model
library(e1071) 
# Divide into test and training
set.seed(123)
test_indis <- sample(1:nrow(heart), 1/3*nrow(heart))
test <- heart[test_indis, ]
training <- heart[-test_indis, ]

# SVM with a linear kernel
tune.model <- tune(svm, target~., data = training, kernel = "linear",
                   ranges = list(cost = c(0.001, 0.01, 0.05, 1, 5, 10, 100)))
tune.model
summary(tune.model)

bestmod <- tune.model$best.model
bestmod

# predict the test data
y_hat <- predict(bestmod, newdata = test)
y_true <- test$target
svm.pred <-ifelse(y_hat>0.5,1,0)
table(predict = svm.pred, truth = y_true)
mean(svm.pred==y_true)
