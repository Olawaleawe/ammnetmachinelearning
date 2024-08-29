#############Using ADASYN FOR HANDLING DATA IMBAALANCE#########
###############################################################
############AUTHOR: OLAWALE AWE, PhD#####################

#ADASYN (Adaptive Synthetic Sampling) is a 
#more advanced version of SMOTE,
#which also generates synthetic data but focuses on the harder-to-learn samples.

#In R, ADASYN can be implemented using the ADASYN function from the UBL package. 

#Here's how you can perform ADASYN in R:

### 1. Install and Load the Required Packages*
library(ROSE)
library(smotefamily)
library(UBL)
library(imbalance)
### 2. Load and Prepare Your Data

# Load the health (malaria) data
mdata = read.csv("Malaria-Data.csv", header = TRUE)
dim(mdata)
mdata
head(mdata)
names(mdata)
attach(mdata)
mdata$severe_maleria =as.factor(mdata$severe_maleria)

### 3. Check the imbalance in the data
table(mdata$severe_maleria)
prop.table(table(mdata$severe_maleria))

#Similar to SMOTE, make sure your dataset is ready for ADASYN, with the target variable as a factor.
mdata$severe_maleria = as.factor(mdata$severe_maleria)
# 3. Apply ADASYN
#Now, apply the ADASYN function to your dataset.

set.seed(123)
Ada1bal <- AdasynClassif(factor(severe_maleria)~.,data= mdata)

### 4. Inspect the Result
#You can now check the balance of your new dataset:
table(Ada1bal$severe_maleria)

### 5. Inspect the plot of  your Result
plot(Ada1bal$severe_maleria)

#Using ADASYN can be particularly useful when you have a highly imbalanced dataset,
#and you want to ensure that the classifier pays more attention to the difficult-to-learn minority class instances.

#SMOTE
#You can use ROSE for performing SMOTE
Smotedata=ROSE(severe_maleria~., data=mdata, seed=123)$data
plot(Smotedata$severe_maleria, ylim=c(0, 250))
box()
