# Car-Price-Predictor

## Overview

This project is based on a Kaggle competition proposed in data analytics class where the goal is to predict the price of a diamond based on their features, using a provided [dataset](https://www.kaggle.com/c/diamonds0819/data) and applying differents agorithms of machine learning find the result which fits with a hidden answer stored in Kaggle.

## The Project

In this project two datsets are provided: One with the answer that will be used to train our model, and another without answer that will be used to apply our model trained and make a prediction to be submitted to kaggle.

### Structure and Files

This project is structured in three differents folders:

* __input.-__ In this folder we will allocate both datasets(data.csv and test.csv), and after a first data processing process, two new files will be generated and allocated there, with our data clean and ready to be used to fit an predict with the differents models.


* __output.-__ In this folders two files will be allocated. A first file with a log of the results from the differents cleaning process in different ways and the result of the RMSE (root mean squared error) provided by the differents models used and their modifications. It will be used as a guide to select the best model with the best hypesr-paramenters configuration. A second file called Submission.csv with the prediction to be submitted to Kaggle.

* __src.-__ In this folder the differents scripts will be allocated.

### Scripts

#### Data Procesing

The first step is to process all the data, clean them and deliver it ready to be used afterwards in our model fitting and predictions. This algorithm tranforms both dataset in the same way in order to fit and precdict the models in the ame way always. SOme columns were removed, som were transfomed, "get dummies" pandas method is applied to categorical data in order to use them in our predicitions and the numerical data were standarized in order to give the same "weight" to all columns when fitting the model.
It delivers two files to our __input__ folder that will be used by every model fitting script and appends a ney line to our log with the columns that will remain after the dat celaning process.

#### Models

Is this case, six differents types of models were used to try to fit the data and predict over our test dataset.
Each script takes our cleaned data generated, fits certain model with certain hyper-parameters and predicts over our test set gererating a submission file. Afterwards appends the model, the RMSE and the hyper-parameters used to our log, in oder to study which model fits better and make the submission with the best accuracy as possible.

