# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

# Imports
import numpy as np
import pandas as pd

# Reading the dataset from data
'''
# Concrete Dataset
dataset = pd.read_csv(r'D:\Repositories\linear_regression\data\Concrete_Data_Yeh.csv')
x_labels = ['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']
y_lables = ['csMPa']
x_labels_categorical = []
'''

# 50_Startups dataset
dataset = pd.read_csv(r'D:\Repositories\linear_regression\data\50_Startups.csv')
x_labels = ['R&D Spend', 'Administration', 'Marketing Spend']
y_lables = ['Profit']
x_labels_categorical = ['State']

# Converting categorical data into one hot encoder
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
X_categorical_columns = None
for categorical_column in x_labels_categorical:
    X_categorical = dataset[categorical_column].values.ravel()
    le = LabelEncoder()
    le.fit(X_categorical)
    X_categorical = le.transform(X_categorical)
    X_categorical = X_categorical.reshape(-1,1)
    ohe = OneHotEncoder()
    ohe.fit(X_categorical)
    X_categorical = ohe.transform(X_categorical).toarray()
    if X_categorical_columns is None:
        X_categorical_columns = X_categorical
    else:
        X_categorical_columns = np.concatenate((X_categorical_columns,X_categorical),axis=1)
    
# Creating Dependent and Independent variables
if len(x_labels_categorical) != 0:
    X = np.concatenate((dataset[x_labels].values,X_categorical_columns),axis=1)
else:
    X = dataset[x_labels].values
    
y = dataset[y_lables].values

# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y, test_size = 0.8)

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

# Importing the linear model from sklearn framework
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X = X_train, y = y_train)

#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

# Predicting the Results
y_pred = lr.predict(X_test)

#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#