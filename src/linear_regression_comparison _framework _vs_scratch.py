# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Reading the dataset from data
dataset = pd.read_csv(r'..\\data\\auto_insurance.csv')

# Creating Dependent and Independent variables
X = dataset['X'].values
y = dataset['Y'].values

# Visualizing the data 
title='Linear Regression on <Dataset>'
x_axis_label = 'X-value < The corresponding attribute of X in dataset >'
y_axis_label = 'y-value < The corresponding attribute of X in dataset >'
plt.scatter(X,y)
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()


# Splitting the data into training set and test set
# This splitting can be done with scikit learns test train split or manully by below code
X_train,X_test = np.split(X,indices_or_sections = [int(len(X)*0.8)])
y_train,y_test = np.split(y,indices_or_sections = [int(len(X)*0.8)])

# Reshaping the numpy arrays since the scikit learn model expects 2-D array in further code
X_train_fw = np.reshape(X_train,newshape = (-1,1))
y_train_fw = np.reshape(y_train,newshape = (-1,1))
X_test_fw = np.reshape(X_test,newshape = (-1,1))
y_test_fw = np.reshape(y_test,newshape = (-1,1))

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

# Computing the values of sigma
sigma_X = sum(X_train)
sigma_y = sum(y_train)
sigma_xy = sum(np.multiply(X_train,y_train))
sigma_X_square = sum(np.square(X_train))
n = len(X_train)

# Computing the values of slope and intercept 
m_numerator = (n*sigma_xy)-(sigma_X*sigma_y)
m_denominator =  n*sigma_X_square - math.pow(sigma_X,2)
m = m_numerator/m_denominator

c_numerator = (sigma_y*sigma_X_square)-(sigma_xy*sigma_X)
c_denominator = (n*sigma_X_square) - math.pow(sigma_X,2)
c = c_numerator/c_denominator

# Importing the linear model from sklearn framework
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X = X_train_fw, y = y_train_fw)

#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

# Predicting the Results
y_pred_stat = X_test*m + c
y_pred_fw = lr.predict(X_test_fw)

# Visualizing the Results

# Plotting each result individually
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_pred_fw,c='cyan',label='framework')
plt.scatter(X,y)
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()

plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_pred_stat,c='green',label='statistical formula')
plt.scatter(X,y)
plt.title(title)
plt.legend()
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()

# Combining the results 
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_pred_fw,c='cyan',label='framework')
plt.plot(X_test,y_pred_stat,c='green',label='statistical formula')
plt.scatter(X,y)
plt.title(title)
plt.legend()
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()


#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#