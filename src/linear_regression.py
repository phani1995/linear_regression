# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset from data
dataset = pd.read_csv(r'data\slr09.csv')

#Creating Dependent and Independent variables
dataset = dataset.sort_values('X')
X = dataset['X'].values
y = dataset['Y'].values

#Visualizing the data 
plt.scatter(X,y)
plt.xlabel('pH of well water')
plt.ylabel('Bicarbonate (ppm)')
plt.show()

#Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y, test_size = 0.8)

# Reshaping the numpy arrays since the scikit learn model expects 2-D array in further code
X_train = np.reshape(X_train,newshape = (-1,1))
y_train = np.reshape(y_train,newshape = (-1,1))
X_test = np.reshape(X_test,newshape = (-1,1))
y_test = np.reshape(y_test,newshape = (-1,1))

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

#Importing the linear model from sklearn framework
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X = X_train, y = y_train)

#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

#Predicting the Results
y_pred = lr.predict(X_test)

#Visualizing the Results
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_pred)
plt.xlabel('pH of well water')
plt.ylabel('Bicarbonate (ppm)')
plt.show()
#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#