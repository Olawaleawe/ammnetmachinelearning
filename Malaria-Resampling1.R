#################################################################################
######AMMNET Training on Machine Learning-Based Models for Malaria Prediction####
#####################################################################
#################INSTRUCTOR:O.OLAWALE AWE, PhD.############################
#####################################################################
#############################################################################################
####Install Packages and Libraries############
#install.packages('caret', dependences=TRUE)
#install.packages('tidyverse', dependences=TRUE)
### Load Some Necessary Packages and Libraries
##############################################################
##Load libraries
library(caret) #for training machine learning models
library(psych) ##for description of  data
library(ggplot2) ##for data visualization
library(caretEnsemble)##enables the creation of ensemble models
library(tidyverse) ##for data manipulation
library(mlbench)  ## for benchmarking ML Models
library(flextable) ## to create and style tables
library(mltools) #for hyperparameter tuning
library(tictoc) #for determining the time taken for a model to run
library(ROSE)  ## for random oversampling
library(smotefamily) ## for smote sampling
library(ROCR) ##For ROC curve

#Introduction
#The most popular data science methodologies come from the field 
#of Machine learning.

#Machine learning (ML) is a subfield of artificial intelligence (AI) that focuses 
#on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. 
#Instead, these models learn from data and improve their performance over time. 

#Prediction problems can be divided into categorical and continuous outcomes.

#In supervised machine learning, data in the form of:
  
#  1. The outcome that we want to predict and 
#2. The feature variables that we will use to predict the outcome

####Load the Data you want to work on 

######Load the Malaria data given
mdata = read.csv("Malaria-Data.csv", header = TRUE)
dim(mdata)
mdata
head(mdata)
names(mdata)
#str(odata)
attach(mdata)
summary(mdata) ###Descriptive Statistics
#describe(mdata)###Descriptive Statistics
sum(is.na(mdata))###Check for missing data

###Note: For the purpose of this training, 
#it is assumed that the data is already clean and preprocessed 
#Use the train option na.action = na.pass if you will 
#be imputing missing data. 

#Also, use this option when #predicting new data containing missing values.

###Rename the classes of the Target variable and plot it to determine imbalance
mdata$severe_maleria <- factor(mdata$severe_maleria, 
                           levels = c(0,1), 
                           labels = c('Not Severe', 'Severe'))
###Plot Target Variable
plot(factor(severe_maleria), names= c('Not Severe', 'Severe'), col=c(2,3), ylim=c(0, 600), ylab='Respondent', xlab='Malaria Diagnosis')
box()
#Or use ggplot 
ggplot(mdata, aes(x = factor(severe_maleria))) + geom_bar() + labs(x = "Malaria Detected", y = "Count")

##### Now Let's train some machine learning models using package caret
#The caret R package (Kuhn et al. 2007) (short for Classification And REgression Training) 
#to carry out machine learning tasks in RStudio.

#The caret package offers a range of tools and models for classification and regression machine learning problems.

#In fact, it offers over 200 different machine learning models from which to choose. 
#Don’t worry, we don’t expect you to use them all!

###VIEW THE AVAILABLE MODELS IN CARET
models= getModelInfo()
names(models)
################################################################
#TODAY we are going to buld the following 10 machine learning models:
#SVM
#RANDOM FOREST
#NAIVE BAYES
#LR
#KNN
#LDA
#NNET
#LVQ
#Bagging
#Boosting

#STEPS
#1. Data Preparation and Preprocessing
# Cleaning, Feature Engineering,Visualization, Data Splitting, etc
#2. Define the Training Control- Set up cross validation
#3. Train the Models- Select the ML models you want to train 
#4. Evaluate your model using test data
#5. Tune the hyperparameters (optional)

###DATA PARTITION FOR MACHINE LEARNING
##################################################################
ind=sample(2, nrow(mdata),replace =T, prob=c(0.70,0.30))
train=mdata[ind==1,]
test= mdata[ind==2,]
#Get the dimensions of your train and test data
dim(train)
dim(test)
#################################################################
####Prepare training scheme for cross-validation#################
#################################################################
control <- trainControl(method="repeatedcv", number=10, repeats=5)
control= trainControl(method='cv', )
##################################################################

#####TRAIN YOUR ML MODELS
# Train SVM model
set.seed(123)
tic()
SvmModel <- train(factor(severe_maleria)~., data=train, method="svmRadial", trControl=control)
toc()
SvmModel
Svmpred= predict(SvmModel,newdata = test)
SVM.cM<- confusionMatrix(Svmpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
SVM.cM
#plotting confusion matrix
SVM.cM$table
fourfoldplot(SVM.cM$table, col=rainbow(4), main="Imbalanced SVM Confusion Matrix")
plot(varImp(SvmModel, scale=T))
#####################################################################
# Train Random Forest model

set.seed(123)
tic()
RFModel <- train(factor(severe_maleria)~., data=train, method="rf", trControl=control)
toc()
RFModel
RFpred=predict(RFModel,newdata = test)
RF.cM<- confusionMatrix(RFpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')

#plotting confusion matrix
RF.cM$table
fourfoldplot(RF.cM$table, col=rainbow(4), main="Imbalanced RF Confusion Matrix")
plot(varImp(RFModel, scale=T))
###################################################################

# Train an Logisitic Regression model
set.seed(123)
lrModel <- train(factor(severe_maleria)~., data=train, method="glm", trControl=control)
lrModel
lrpred=predict(lrModel,newdata = test)
lr.cM<- confusionMatrix(lrpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
#plotting confusion matrix
lr.cM$table
fourfoldplot(lr.cM$table, col=rainbow(4), main="Imbalanced LR Confusion Matrix")
plot(varImp(lrModel, scale=T))
##############################################################
# Train k- Nearest Neigbour model
set.seed(123)
knnModel <- train(factor(severe_maleria)~., data=train, method="knn", preProc=c("center", "scale"), trControl=control)
knnModel
knnpred=predict(knnModel,newdata = test)
knn.cM<- confusionMatrix(knnpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
knn.cM
#plotting confusion matrix
knn.cM$table
fourfoldplot(knn.cM$table, col=rainbow(4), main="Imbalanced KNN Confusion Matrix")
plot(varImp(knnModel, scale=T))
plot(age)
##############################################################
# Train Neural Net model
set.seed(123)
nnModel <- train(factor(severe_maleria)~., data=train, method="nnet", trControl=control)
nnModel
nnpred=predict(nnModel,newdata = test)
nn.cM<- confusionMatrix(nnpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
nn.cM
#plotting confusion matrix
nn.cM$table
fourfoldplot(nn.cM$table, col=rainbow(4), main="Imbalanced NN Confusion Matrix")
plot(varImp(nnModel, scale=T))
#############################################################
# Train Naive Bayes model
set.seed(123)
nbModel <- train(factor(severe_maleria)~., data=train, method="nb", trControl=control)
nbModel
nbpred=predict(nbModel,newdata = test)
nb.cM<- confusionMatrix(nbpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
#plotting confusion matrix
nb.cM$table
fourfoldplot(nb.cM$table, col=rainbow(4), main="Imbalanced NB Confusion Matrix")
plot(varImp(nbModel, scale=T))
####################################################################
#Train Linear Discriminant Analysis model
set.seed(123)
ldaModel <- train(factor(severe_maleria)~., data=train,preProc=c("center", "scale"), method="lda", trControl=control)
ldaModel
ldapred=predict(ldaModel,newdata = test)
lda.cM<- confusionMatrix(ldapred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
#plotting confusion matrix
lda.cM$table
fourfoldplot(lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")
plot(varImp(ldaModel, scale=T))
######################################################################
#Train Linear Vector Quantization model
names(mdata)
mdata
set.seed(123)
lvqModel <- train(factor(severe_maleria)~., data=train, method="lvq", preProc=c("center", "scale"), trControl=control)
lvqModel
lvqpred=predict(lvqModel,newdata = test)
lvq.cM<- confusionMatrix(lvqpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
#plotting confusion matrix
lvq.cM$table
fourfoldplot(lvq.cM$table, col=rainbow(4), main="Imbalanced LVQ Confusion Matrix")
plot(varImp(lvqModel, scale=T))
########################################################################
# Train Bagging model
set.seed(123)
bagModel <- train(factor(severe_maleria)~., data=train, method="treebag", trControl=control)
bagModel
bagpred=predict(bagModel,newdata = test)
bag.cM<- confusionMatrix(bagpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
bag.cM
#plotting confusion matrix
bag.cM$table
fourfoldplot(bag.cM$table, col=rainbow(4), main="Imbalanced Bagging Confusion Matrix")
plot(varImp(bagModel, scale=T))
################################################################
# Train Boosting model
set.seed(123)
boModel <- train(factor(severe_maleria)~., data=train, method="ada",preProc=c("center", "scale"), trControl=control)
boModel
bopred=predict(boModel,newdata = test)
bo.cM<- confusionMatrix(bopred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
bo.cM
#plotting confusion matrix
bo.cM$table
fourfoldplot(bo.cM$table, col=rainbow(4), main="Imbalanced Boosting Confusion Matrix")
plot(varImp(boModel, scale=T))
############################### Plot/TABULATE YOUR RESULTS#########################################

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
                          Boosting=boModel))
#####################################################################
## summarize the distributions of the results 
summary(results)
## bwplots of results
bwplot(results,  main='Comparison of Models')
## dot plots of results

bwplot(results, layout = c(1, 2), scales = "free", as.table = TRUE)
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
####CREATE ROC curve for your models-try for all models
# Make predictions on the test set using type='prob'
predrf <- predict(RFModel, newdata = test, type = "prob")
# Create a prediction object needed by ROCR
pred_rf <- prediction(predrf[, "Severe"], test$severe_maleria)
# Calculate performance measures like ROC curve
perf_rf <- performance(pred_rf, "tpr", "fpr")
# Plot the ROC curve
plot(perf_rf, colorize = TRUE, main = "ROC Curve-Random Forest")
# Compute AUC
auc_value <- performance(pred_rf, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position
#################################################################

####CREATE ROC curve for KNN
##################################################################
#library(ROCR)
# Make predictions on the test set using type='prob'
predknn <- predict(knnModel, newdata = test, type = "prob")
# Create a prediction object needed by ROCR
pred_knn <- prediction(predknn[, "Severe"], test$severe_maleria)
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
pred_lr <- prediction(predlr[, "Severe"], test$severe_maleria)
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
#AUC-ROC for LVQ
##################################################################
# Make predictions on the test set using type='prob'
predlr <- predict(lrModel, newdata = test, type = "prob")
# Load the ROCR package
library(ROCR)
# Create a prediction object needed by ROCR
pred_lr <- prediction(predlr[, "Severe"], test$severe_maleria)
# Calculate performance measures like ROC curve
perf_lr <- performance(pred_lr, "tpr", "fpr")
# Plot the ROC curve
plot(perf_lr, colorize = TRUE, main = "ROC Curve-LVQ")
# Compute AUC
auc_value <- performance(pred_lr, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position and other text parameters as needed
##############################################################
#Try for other models

##HANDLING DATA IMBALANCE
################ The Problem of Imbalanced Data in Medical Science########################################

#Imbalanced data is a common issue in medical science.

#It can significantly impact the performance and reliability of machine learning models.

#Imbalanced data occurs when the number of instances of one class far exceeds the number of instances of another class.

#In medical applications, this often manifests as rare conditions or diseases being underrepresented in the dataset compared to more common conditions.

#Failing to accurately diagnose or predict rare conditions due to imbalanced data can lead to adverse patient outcomes,
#poor diagnosis, loss of trust in medical AI systems, and potential legal issues.

#Potential strategies to address these challenges include resampling the dataset and using ensemble models.

#Resampling Methods include Oversampling, Undersampling and Hybrid Sampling

## We shal explore these methods in this Demo.

#######################################################################
#### Oversampling 
#####################################################################
# Create Oversampled data --------------------------------------------------------
 over <- ovun.sample(factor(severe_maleria)~., data = train, method = "over")$data
 over
 prop.table(table(over$severe_maleria))
 plot(over$severe_maleria, ylim=c(0,400),col=c('red','blue'))
 # Model building ----------------------------------------------------------
 # prepare training scheme for cross-validation
 control <- trainControl(method="repeatedcv", number=10, repeats=5)
  # Train SVM model
 set.seed(123)
 over.svmModel <- train(factor(severe_maleria)~., data=over, method="svmRadial", trControl=control)
 over.svmModel
 over.svmpred=predict(over.svmModel,newdata = test)
 over.SVM.cM<- confusionMatrix(over.svmpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
 over.SVM.cM
  #plotting confusion matrix
 over.SVM.cM$table
 fourfoldplot(over.SVM.cM$table, col=rainbow(4), main="Oversampled SVM Confusion Matrix")
 ##################################################################
 ##################################################################
 
  ## Train Random Forest model
 set.seed(123)
 over.RFModel <- train(factor(severe_maleria)~., data=over, method="rf", trControl=control)
 over.RFModel
 over.RFpred=predict(over.RFModel,newdata = test)
 over.RF.cM<- confusionMatrix(over.RFpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
 over.RF.cM
 #plot confusion matrix
 over.RF.cM$table
 fourfoldplot(over.RF.cM$table, col=rainbow(4), main="Oversampled RF Confusion Matrix")
 
 ## Train Logisitic Regression model
 set.seed(123)
 over.lrModel <- train(factor(severe_maleria)~., data=over, method="glm", trControl=control)
 over.lrModel
 over.lrpred=predict(over.lrModel,newdata = test)
 over.lr.cM <- confusionMatrix(over.lrpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
 over.lr.cM
 #plot confusion matrix
 over.lr.cM$table
 fourfoldplot(over.lr.cM$table, col=rainbow(4), main="Oversampled LR Confusion Matrix")
 
 ## Train k- Nearest Neigbour model
 set.seed(123)
 over.knnModel <- train(factor(severe_maleria)~., data=over, method="knn", trControl=control)
 over.knnModel
 over.knnpred=predict(over.knnModel,newdata = test)
 over.knn.cM <- confusionMatrix(over.knnpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
 over.knn.cM
 #plot confusion matrix
 over.knn.cM$table
 fourfoldplot(over.knn.cM$table, col=rainbow(4), main="Oversampled KNN Confusion Matrix")
 

 
 
 
 
 ############################################
 #Collect all resamples and compare
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
 
 ###uNDERSAMPLING
 #################################################################
------------------------------------------------------
 under = ovun.sample(factor(severe_maleria)~., data = train, method = "under")$data
 under
 
 
 