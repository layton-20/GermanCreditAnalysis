# GermanCreditAnalysis
This project takes a data set of credit and banking information and predicts the classification of each record as having a good credit score or not. 
---
title: "MIS 510 Portfolio Project Option 1"
author: "Olivia Layton"
date: "August 14, 2019"
output: word_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
# Reads CSV into dataframe, specifies that the first row is a header row 

Credit.df <- read.csv("GermanCredit.csv", header = TRUE)

# Exploratory Functions

## Opens dataframe in a new window
View(Credit.df)

## Removes Observation column from dataframe
Credit.df <- Credit.df[-c(1)]

## Summary Statistics
summary(Credit.df)

## Counts cases of good credit (700) and bad credit (300)
library(plyr)
count(Credit.df, "RESPONSE")

## Counts number of missing values (0)
sum(is.na(Credit.df))

## Calculates mean of Credit Amount (3271.258)
mean(Credit.df$AMOUNT)


# Data Dimension Reduction

## Creates a correlation matrix for Credit.df
CorrelationMatrix.df <- round(cor(Credit.df),2)

## Saves correlation matrix as dataframe
CorrelationMatrix.df <- as.data.frame(CorrelationMatrix.df)

## Opens CorrelationMatrix.df in new window
View(CorrelationMatrix.df)

## Identifies variables that have a correlation of an absolute value less than 0.05 in the Response column

CorrelationMatrix.df[abs(CorrelationMatrix.df$RESPONSE) < 0.05, ]

## Remove the columns with correlations of absolute values less than 0.05
CreditDim.df <- Credit.df[-c(6,9,16,19,27,28,29)]

## Opens CreditDim.df in new window
View(CreditDim.df)

# Partitioning into traning and validation data

## Sets seed so that the same samples are taken for each group every time this is run
set.seed(1)

## Assigns 80% of CreditDim data to training rows
train.rows <- sample(rownames(CreditDim.df), dim(CreditDim.df)[1]*0.8)

## Assigns 20% of CreditDim data to validation rows
valid.rows <- sample(setdiff(rownames(CreditDim.df), train.rows), dim(CreditDim.df)[1]*0.2)

## Assigns training rows to training data set and validation rows to validation data set
train.data <- CreditDim.df[train.rows, ]
valid.data <- CreditDim.df[valid.rows, ]

## Counts records in each set (Gives dimensions, which specify number of rows and columns)
dim(train.data)
dim(valid.data)

# Logistic Regression using training data

## Runs logistic regression using RESPONSE as the output variable, train.data as the data, and specifying that the output is in binomial format
logit.reg <- glm(RESPONSE ~ ., data = train.data, family = "binomial")
options(scipen=999)
summary(logit.reg)

## Use Gains library

library(gains)

## Calculate predicted values and gains using probability type
pred <- predict(logit.reg, train.data, type="response")
gain <- gains(train.data$RESPONSE, pred)

## Calculate predicted values for validation data
predvalid <- predict(logit.reg, valid.data, type="response")


## Plot predicted values vs. gains
plot(c(0,gain$cume.pct.of.total*sum(train.data$RESPONSE))~c(0,gain$cume.obs),xlab="Records Accumulated", ylab = "Cumulative Value of Good Credit Records", main="Lift Chart for Response Variable (Good Credit)", type='l')
lines(c(0,sum(train.data$RESPONSE))~c(0,dim(train.data)[1]), lty=2)

## Uses Caret library

library(caret)

## Sends pred and predvalid value to 1 if probability is greater than 0.785, and 0 otherwise. 
pred <- ifelse(pred>0.785,1,0)
predvalid <- ifelse(predvalid>0.785,1,0)

class(pred)
class(train.data$RESPONSE)
class(valid.data$RESPONSE)

## Turns pred, predvalid, valid.data$Response and train.data$Response into factors

pred <- as.factor(pred)
predvalid <- as.factor(predvalid)
train.data$RESPONSE <- as.factor(train.data$RESPONSE)
valid.data$RESPONSE <- as.factor(valid.data$RESPONSE)



## Confusion Matrices for train.data and valid.data
confusionMatrix(pred, train.data$RESPONSE)

confusionMatrix(predvalid, valid.data$RESPONSE)




View(train.data)

# Create classification tree
## Use rpart to make a classification tree with RESPONSE as the output factor, variable predictors, classification, the training data, cp of 0.009, maximum depth of 15, and minimum records in a terminal node of 1. Use a loss matrix to consider costs of misclassification.
library(rpart.plot)
fit <- rpart(RESPONSE~., 
method="class",
data=train.data,
parms=list(split="gini", loss=matrix(c(0,100,1000,0), byrow=TRUE, nrow=2)),
control=rpart.control(minbucket=1, cp=0.009, maxdepth=15))


# Plot classification tree
## Plots the classification tree using type 0 tree, type 8 for details for node labels, truncates variable length to 6, and increases character size by 30%.

prp(fit, type = 0, extra = 8,  varlen = 6, tweak = 1.3, uniform=TRUE)

# Confusion Matrix
## Predicts and creates a confusion matrix for train.data$Response based on classification tree

fit.pred <- predict(fit, train.data, type="class")

confusionMatrix(fit.pred, train.data$RESPONSE)

## Predicts and creates a confusion matrix for valid.data$Response based on classification tree

fit.predvalid <- predict(fit, valid.data, type="class")

confusionMatrix(fit.predvalid, valid.data$RESPONSE)
