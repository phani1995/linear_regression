# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset from data
dataset = pd.read_excel(r'data\\slr09.xls')

#Creating Dependent and Independent variables
X = dataset['X'].values
y = dataset['Y'].values

# Reshaping the numpy arrays since the scikit learn model expects 2-D array in further code
X = np.reshape(X,newshape = (-1,1))
y = np.reshape(y,newshape = (-1,1))
#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

#Importing the linear model from sklearn framework
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X = X, y = y)
#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#
y_pred = lr.predict(X)
plt.plot(X,y_pred)

#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#