#################### RESAMPLING METHODS FOR HANDLING HEALTH DATA WITH R ###############################
####################### AUTHOR: OLAWALE AWE, PhD.######################################################
########################Email: olawaleawe@gmail.com ###################################################
#######################################################################################################

############ The Problem of Imbalanced Data in Medical Science########################################

#Imbalanced data is a common issue in medical science.

#It can significantly impact the performance and reliability of machine learning models.

#Imbalanced data occurs when the number of instances of one class far exceeds the number of instances of another class.

#In medical applications, this often manifests as rare conditions or diseases being underrepresented in the dataset compared to more common conditions.

#Failing to accurately diagnose or predict rare conditions due to imbalanced data can lead to adverse patient outcomes,
#poor diagnosis, loss of trust in medical AI systems, and potential legal issues.

#Potential strategies to address these challenges include resampling the dataset and using ensemble models.

#Resampling Methods include Oversampling, Undersampling and Hybrid Sampling

## We shal explore these methods in this Demo.


####Install Packages and Libraries
#intall.packages('caret', dependences=TRUE)
### Load Some Necessary Packages and Libraries
library(caretEnsemble)
library(tidyverse)
library(mlbench)
library(caret)
library(flextable)
library(mltools) 
library(tictoc)
library(ROSE)
#library(smotefamily)
library(ROCR)
####################################################################
####################################################################
###Determine your working directory
getwd()
#setwd()
####################################################################
##DATA
#####Data Source: https://ghdx.healthdata.org/record/south-africa-national-health-and-nutrition-examination-survey-2012
##South Africa National Health and Nutrition Examination Survey 2012 (SANHANES)
###Contains predictors of Overweight and Obesity among South Africans
### Data was analyzed in this article below: 
##Awe, O. O., Dukhi, N., & Dias, R. (2023). Shrinkage heteroscedastic discriminant algorithms for classifying multi-class high-dimensional data: Insights from a national health survey. Machine Learning with Applications, 12, 100459.

###Load the Data
odata = read.csv("Odata.csv", header = TRUE)
dim(odata)
odata
head(odata)
names(odata)
#str(odata)
attach(odata)
summary(odata) ###Descriptive Statistics
#describe(odata)###Descriptive Statistics
sum(is.na(odata))###Check for missing values

###Rename the classes of the Target variable and plot it to determine data imbalance
odata$overweight <- factor(odata$overweight, 
                           levels = c(0,1), 
                           labels = c('Normal', 'Obese'))
###Plot Target Variable
plot(factor(overweight), names= c('Normal', 'Obese'), col=c(2,3), ylim=c(0, 600), ylab='Respondent', xlab='BMI Category')
box()
##Class Imbalance
prop.table(table(odata$overweight))
###Assuming that all the EDA and feature selection has been performed.
################################################################
###
#Perform Featureplot to see the data distribution at a glance
featurePlot(x = odata[, -which(names(odata) == "overweight")],   # Predictors
            y = odata$overweight,                               # Target variable
            plot = "box",                                       # Type of plot (e.g., "box", "density", "scatter")
            strip = strip.custom(strip.names = TRUE),            # Add strip labels
            scales = list(x = list(relation = "free"),           # Scales for x-axis
                          y = list(relation = "free")))          # Scales for y-axis
#####Pairs Plot
#plot(odata)
#pairs(odata)
################################################################

###SEE AVAILABLE MODELS IN CARET
models= getModelInfo()
names(models)

###DATA PARTITION
##################################################################
ind=sample(2, nrow(odata),replace=T, prob=c(0.70,0.30))
train=odata[ind==1,]
test= odata[ind==2,]
#Get the dimensions of your train and test test data
dim(train)
dim(test)

# Model Building ----------------------------------------------------------
# prepare training scheme for cross-validation
control <- trainControl(method="repeatedcv", savePredictions=TRUE, number=10, repeats=5)

#####TRAIN YOUR MODELS
# Train SVM model
set.seed(123)
tic()
SvmModel <- train(factor(overweight)~., data=train, method="svmRadial", preProc=c("center", "scale"), trControl=control, na.action = na.omit)
toc()
SvmModel
Svmpred= predict(SvmModel,newdata = test)
SVM.cM<- confusionMatrix(Svmpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
SVM.cM
#plot confusion matrix
SVM.cM$table
fourfoldplot(SVM.cM$table, col=rainbow(4), main="Imbalanced SVM Confusion Matrix")
plot(varImp(SvmModel, scale=T))

# Train Random Forest model
set.seed(123)
tic()
RFModel <- train(factor(overweight)~., data=train, method="rf", preProc=c("center", "scale"), trControl=control)
toc()
RFModel
RFpred=predict(RFModel,newdata = test)
RF.cM<- confusionMatrix(RFpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plot confusion matrix
RF.cM$table
fourfoldplot(RF.cM$table, col=rainbow(4), main="Imbalanced RF Confusion Matrix")
plot(varImp(RFModel, scale=T))

# Train Logistic Regression model
set.seed(123)
lrModel <- train(factor(overweight)~., data=train, method="glm", preProc=c("center", "scale"),trControl=control)
lrModel
lrpred=predict(lrModel,newdata = test)
lr.cM<- confusionMatrix(lrpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plot confusion matrix
lr.cM$table
fourfoldplot(lr.cM$table, col=rainbow(4), main="Imbalanced LR Confusion Matrix")
plot(varImp(lrModel, scale=T))
####################################################################
# Train k- Nearest Neigbour model
set.seed(123)
knnModel <- train(factor(overweight)~., data=train, method="knn", preProc=c("center", "scale"),trControl=control)
knnModel
knnpred=predict(knnModel,newdata = test)
knn.cM<- confusionMatrix(knnpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plot confusion matrix
knn.cM$table
fourfoldplot(knn.cM$table, col=rainbow(4), main="Imbalanced KNN Confusion Matrix")
plot(varImp(knnModel, scale=T))
############################################################
## Train Neural Net model
set.seed(123)
nnModel <- train(factor(overweight)~., data=train, method="nnet", preProc=c("center", "scale"),trControl=control)
nnModel
nnpred=predict(nnModel,newdata = test)
nn.cM<- confusionMatrix(nnpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plotting confusion matrix
nn.cM$table
fourfoldplot(nn.cM$table, col=rainbow(4), main="Imbalanced NN Confusion Matrix")
plot(varImp(nnModel, scale=T))
####################################################################
##Train Naive Bayes model
set.seed(123)
nbModel <- train(factor(overweight)~., data=train, method="nb",preProc=c("center", "scale"), trControl=control)
nbModel
nbpred=predict(nbModel,newdata = test)
nb.cM<- confusionMatrix(nbpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plot confusion matrix
nb.cM$table
fourfoldplot(nb.cM$table, col=rainbow(4), main="Imbalanced NB Confusion Matrix")
plot(varImp(nbModel, scale=T))
####################################################################
##Train Linear Discriminant Analysis model
set.seed(123)
ldaModel <- train(factor(overweight)~., data=train, method="lda", preProc=c("center", "scale"),trControl=control)
ldaModel
ldapred=predict(ldaModel,newdata = test)
lda.cM<- confusionMatrix(ldapred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plot confusion matrix
lda.cM$table
fourfoldplot(lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")
plot(varImp(ldaModel, scale=T))
####################################################################
####################################################################
##Train Linear Vector Quantization model
set.seed(123)
lvqModel <- train(overweight~., data=train, method="lvq", preProc=c("center", "scale"),trControl=control)
lvqModel
lvqpred=predict(lvqModel,newdata = test)
lvq.cM<- confusionMatrix(lvqpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
lvq.cM
#plot confusion matrix
lvq.cM$table
fourfoldplot(lvq.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")
plot(varImp(lvqModel, scale=T))

###################################################################
# Train a Bagging model
set.seed(123)
bagModel <- train(factor(overweight)~., data=train, method="treebag", preProc=c("center", "scale"),trControl=control)
bagModel
bagpred=predict(bagModel,newdata = test)
bag.cM<- confusionMatrix(bagpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plot confusion matrix
bag.cM$table
fourfoldplot(bag.cM$table, col=rainbow(4), main="Imbalanced Bagging Confusion Matrix")
plot(varImp(bagModel, scale=T))
#####################################################################
## Train a Boosting model
set.seed(123)
boModel <- train(factor(overweight)~., data=train, method="ada",preProc=c("center", "scale"), trControl=control)
boModel
bopred=predict(boModel,newdata = test)
bo.cM<- confusionMatrix(bopred,as.factor(test$overweight), positive = 'Obese', mode='everything')
#plot confusion matrix
bo.cM$table
fourfoldplot(bo.cM$table, col=rainbow(4), main="Imbalanced Boosting Confusion Matrix")
plot(varImp(boModel, scale=T))
############################### TABULATE/PLOT RESULTS #########################################

# collect all resamples and compare models
results <- resamples(list(SVM=SvmModel,
                          Bagging=bagModel,
                          LR=lrModel,
                          NB=nbModel,
                          RF=RFModel,
                          KNN= knnModel,
                          NN= nnModel,
                          LDA =ldaModel, 
                          LVQ= lvqModel,
                          Boosting=boModel
))
#####################################################################
## summarize the distributions of the results 
summary(results)
## bwplots of results
bwplot(results,  main='Comparison of Models')
## dot plots of results
dotplot(results)

##############################################################
#Kappa Statistic
#One of the primary advantages of kappa is that it accounts for the agreement occurring by chance.
#This is particularly important in imbalanced datasets where a high accuracy might be misleading. 
#For instance, in a dataset with 90% negatives and 10% positives, a model that always predicts negative would have 90% accuracy but 
#𝜅≈ 0
#Kappa provides a standardized metric for comparing the performance of different models or classifiers, regardless of the underlying class distribution. 
#This is especially useful when dealing with multiple models or datasets.

#In addition to accuracy, precision, recall, and F1-score, kappa provides another dimension to evaluate model performance.
#It helps in understanding the reliability and robustness of the model's predictions.


####################AUC_ROC CURVES############################
##############################################################
####CREATE ROC curve for KNN
##################################################################
#library(ROCR)
# Make predictions on the test set using type='prob'
predknn <- predict(knnModel, newdata = test, type = "prob")
# Create a prediction object needed by ROCR
pred_knn <- prediction(predknn[, "Obese"], test$overweight)
# Calculate performance measures like ROC curve
perf_knn <- performance(pred_knn, "tpr", "fpr")
# Plot the ROC curve
plot(perf_knn, colorize = TRUE, main = "ROC Curve of KNN")
# Compute AUC
auc_value <- performance(pred_knn, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position and other text parameters as needed
###################################################################
##################################################################
####CREATE ROC curve for LR
##################################################################
# Make predictions on the test set using type='prob'
predlr <- predict(lrModel, newdata = test, type = "prob")
# Create a prediction object needed by ROCR
pred_lr <- prediction(predlr[, "Obese"], test$overweight)
# Calculate performance measures like ROC curve
perf_lr <- performance(pred_lr, "tpr", "fpr")
# Plot the ROC curve
plot(perf_lr, colorize = TRUE, main = "ROC Curve for LR")
# Compute AUC
auc_value <- performance(pred_lr, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position and other text parameters as needed
####################################################################

##############################################################
############ Oversampled data --------------------------------
##############################################################
##############################################################

over <- ovun.sample(factor(overweight)~., data = train, method = "over")$data
over
plot(over$overweight, ylim=c(0,400),col=c('red','blue'))
box()
# Model building ----------------------------------------------------------
# prepare training scheme for cross-validation
control <- trainControl(method="repeatedcv", number=10, repeats=5)
## Train SVM model
set.seed(123)
over.svmModel <- train(factor(overweight)~., data=over, method="svmRadial", trControl=control)
over.svmModel
over.svmpred=predict(over.svmModel,newdata = test)
over.SVM.cM<- confusionMatrix(over.svmpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.SVM.cM
#plot confusion matrix
over.SVM.cM$table
fourfoldplot(over.SVM.cM$table, col=rainbow(4), main="Oversampled SVM Confusion Matrix")

## Train Random Forest model
set.seed(123)
over.RFModel <- train(factor(overweight)~., data=over, method="rf", trControl=control)
over.RFModel
over.RFpred=predict(over.RFModel,newdata = test)
over.RF.cM<- confusionMatrix(over.RFpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.RF.cM
#plot confusion matrix
over.RF.cM$table
fourfoldplot(over.RF.cM$table, col=rainbow(4), main="Oversampled RF Confusion Matrix")

## Train Logisitic Regression model
set.seed(123)
over.lrModel <- train(factor(overweight)~., data=over, method="glm", trControl=control)
over.lrModel
over.lrpred=predict(over.lrModel,newdata = test)
over.lr.cM <- confusionMatrix(over.lrpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.lr.cM
#plot confusion matrix
over.lr.cM$table
fourfoldplot(over.lr.cM$table, col=rainbow(4), main="Oversampled LR Confusion Matrix")

## Train k- Nearest Neigbour model
set.seed(123)
over.knnModel <- train(factor(overweight)~., data=over, method="knn", trControl=control)
over.knnModel
over.knnpred=predict(over.knnModel,newdata = test)
over.knn.cM <- confusionMatrix(over.knnpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.knn.cM
#plot confusion matrix
over.knn.cM$table
fourfoldplot(over.knn.cM$table, col=rainbow(4), main="Oversampled KNN Confusion Matrix")

## Train Neural Net model
set.seed(123)
over.nnModel <- train(factor(overweight)~., data=over, method="nnet", trControl=control)
over.nnModel
over.nnpred=predict(over.nnModel,newdata = test)
over.nn.cM <- confusionMatrix(over.nnpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.nn.cM
#plot confusion matrix
over.nn.cM$table
fourfoldplot(over.nn.cM$table, col=rainbow(4), main="Oversampled NN Confusion Matrix")

## Train Naive Bayes model
set.seed(123)
over.nbModel <- train(factor(overweight)~., data=over, method="nb", trControl=control)
over.nbModel
over.nbpred=predict(over.nbModel,newdata = test)
over.nb.cM <- confusionMatrix(over.nbpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.nb.cM
#plot confusion matrix
over.nb.cM$table
fourfoldplot(over.nb.cM$table, col=rainbow(4), main="Oversampled NB Confusion Matrix")

## Train Linear Discriminant Analysis model
set.seed(123)
over.ldaModel <- train(factor(overweight)~., data=over, method="lda", trControl=control)
over.ldaModel
over.ldapred=predict(ldaModel,newdata = test)
over.lda.cM <- confusionMatrix(ldapred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.lda.cM
##plot confusion matrix
over.lda.cM$table
fourfoldplot(over.lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")
###########################################################################################
## Train Linear Vector Quantization Model
set.seed(123)
over.lvqModel <- train(factor(overweight)~., data=over, method="lvq", trControl=control)
over.lvqModel
over.lvqpred=predict(lvqModel,newdata = test)
over.lvq.cM <- confusionMatrix(lvqpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.lvq.cM
##plot confusion matrix
over.lvq.cM$table
fourfoldplot(over.lvq.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")

#####################################################################
## Train Bagging model
set.seed(123)
over.bagModel <- train(factor(overweight)~., data=over, method="treebag", trControl=control)
over.bagModel
over.bagpred=predict(over.bagModel,newdata = test)
over.bag.cM <- confusionMatrix(over.bagpred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.bag.cM
#plot confusion matrix
over.bag.cM$table
fourfoldplot(over.bag.cM$table, col=rainbow(4), main="Oversampled Bagging Confusion Matrix")

## Train a Boosting model
set.seed(123)
over.boModel <- train(factor(overweight)~., data=over, method="ada", trControl=control)
over.boModel
over.bopred=predict(over.boModel,newdata = test)
over.bo.cM <- confusionMatrix(over.bopred,as.factor(test$overweight), positive = 'Obese', mode='everything')
over.bo.cM 
#plot confusion matrix
over.bo.cM$table
fourfoldplot(over.bo.cM$table, col=rainbow(4), main="Oversampled Boosting Confusion Matrix")
######################################################################


# Collect all resamples and compare
results1 <- resamples(list(SVM=over.svmModel,
                           Bagging=over.bagModel,
                           LR=over.lrModel,
                           NB=over.nbModel,
                           RF=over.RFModel,
                           KNN= over.knnModel,
                           NN= over.nnModel,
                           LDA =over.ldaModel, 
                           LVQ= over.lvqModel,
                           Boosting=over.boModel
))
#####################################################################
## summarize the distributions of the results 
summary(results1)
## boxplots of results
bwplot(results1)
