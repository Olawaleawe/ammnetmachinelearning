#######HYPERPARAMETER TUNING OF DECISION TREES USING caret####
######AUTHOR: OLAWALE AWE, Ph.D. ############################
#############################################################
#############################################################
###LOAD NECESSARY LIBRARIES
library(caret)
library(ROSE)
library(DMwR2)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(ROCR)
library(pROC)
library(tictoc)

###VIEW THE AVAILABLE MODELS IN CARET
models= getModelInfo()
names(models)
###########################################################

###LOAD AND EXAMINE YOUR DATA######
mdata = read.csv("Malaria-Data.csv", header = TRUE, stringsAsFactors = FALSE)
dim(mdata)
mdata
head(mdata)
names(mdata)
#str(odata)
#attach(mdata)
summary(mdata) ###Descriptive Statistics
#describe(mdata)###Descriptive Statistics
sum(is.na(mdata))###Check for missing data
#############################################################

#Feature Plot
#Perform Featureplot to see the data distribution at a glance
featurePlot(x = mdata[, -which(names(mdata) == "severe_maleria")],   # Predictors
            y = mdata$severe_maleria,                               # Target variable
            plot = "box",                                       # Type of plot (e.g., "box", "density", "scatter")
            strip = strip.custom(strip.names = TRUE),            # Add strip labels
            scales = list(x = list(relation = "free"),           # Scales for x-axis
                          y = list(relation = "free")))          # Scales for y-axis

####################################################################
###Rename the classes of the Target variable and plot it to determine imbalance
mdata$severe_maleria <- factor(mdata$severe_maleria, 
                               levels = c(0,1), 
                               labels = c('Not Severe', 'Severe'))
###Plot Target Variable
plot(factor(severe_maleria), names= c('Not Severe', 'Severe'), col=c(2,3), ylim=c(0, 600), ylab='Respondent', xlab='Malaria Diagnosis')
box()
#Or use ggplot 
ggplot(mdata, aes(x = factor(severe_maleria))) + geom_bar() + labs(x = "Malaria Detected", y = "Respondent", fill = 'severe_maleria')+ theme_minimal()

#Data partition
set.seed(123)
trainIndex <- createDataPartition(mdata$severe_maleria, p = 0.75, list = FALSE)
train <- mdata[trainIndex, ]
test <- mdata[-trainIndex, ]
dim(train)
dim(test)

#Ensure your target variable is a factor variable for proper classification
train$severe_maleria <- as.factor(train$severe_maleria)
test$severe_maleria <- as.factor(test$severe_maleria)

#Define the hyperparameter grid
#tuneGrid <- expand.grid(cp = seq(0.001, 0.1, by = 0.01))
# Set up cross-validation
control <- trainControl(method = "cv", number = 10, sampling ='smote')# SMOTE sampling
#control <- trainControl(method = "cv", number = 10, sampling='rose') # Random Oversampling
#control <- trainControl(method = "cv", number = 10, sampling='up') # Oversampling
#control <- trainControl(method = "cv", number = 10, sampling='down') # Undersampling
#control <- trainControl(method = "cv", number = 10)# No sampling

#Train the Decision Tree model USING CARET
set.seed(123)
tic()
dtModel <- train(factor(severe_maleria) ~., data = train, 
                 method = "rpart", 
                 trControl = control, 
                 tuneGrid = tuneGrid_dt)
toc()
#Print the results of the trained model
print(dtModel)
dtModel$results
#Evaluate the model
dtpred=predict(dtModel,newdata = test)
dt.cM<- confusionMatrix(dtpred,as.factor(test$severe_maleria), positive = 'Severe', mode='everything')
dt.cM
#plotting confusion matrix
dt.cM$table
fourfoldplot(dt.cM$table, col=rainbow(4), main="Confusion Matrix of Decision Tree")

### Visualize Variable Importance
plot(varImp(dtModel, scale=T))

#Visualize the Trees
#rpart.plot::rpart.plot(dtModel$finalModel)
rpart.plot(dtModel$finalModel)

#######Create ROC Curve
set.seed(123)
preddt <- predict(dtModel, newdata = test, type = "prob")
# Create a prediction object needed by ROCR
pred_dt<- prediction(preddt[, "Severe"], test$severe_maleria)
# Calculate performance measures like ROC curve
perf_dt <- performance(pred_dt, "tpr", "fpr")
# Plot the ROC curve
plot(perf_dt, colorize = TRUE, main = "ROC Curve-Decision Tree")
# Compute AUC
auc_value <- performance(pred_dt, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position

#Excercise
#Repeat this same process for Random Forest and several other models and compare your results