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
X_train,X_test = np.split(X,indices_or_sections = [int(len(X)*0.8)])
y_train,y_test = np.split(y,indices_or_sections = [int(len(X)*0.8)])

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

def mean_square_error(slope,intercept,x_data,y_data):
    n = len(x_data)
    sum_of_squares = 0
    for i in range(n):
        y_pred = x_data[i]*slope+intercept
        y_actual = y_data[i]
        sum_of_squares += (y_actual - y_pred)**2  
        
    mean_square_error = math.sqrt(sum_of_squares/n)
    return mean_square_error
        
def gradient_slope(slope,intercept,x_data,y_data):
    n = len(x_data)
    sum_of_gradients = 0
    for i in range(n):
        y_pred = x_data[i]*slope + intercept
        y_actual = y_data[i]
        sum_of_gradients += x_data[i]*(y_actual-y_pred)
    sum_of_gradients = -1*sum_of_gradients
    #print("sum of gradients",sum_of_gradients)
    gradient_slope = sum_of_gradients*(2/n)
    #print("The Gradient slope value is",gradient_slope)
    return gradient_slope

def gradient_intercept(slope,intercept,x_data,y_data):
    n = len(x_data)
    sum_of_gradients = 0
    for i in range(n):
        y_pred = x_data[i]*slope +intercept
        y_actual = y_data[i]
        sum_of_gradients += -1 *(y_actual-y_pred)
    sum_of_gradients = -1*sum_of_gradients
    gradient_intercept = sum_of_gradients*(2/n)
    #print("The Gradient Intercept value is ",gradient_intercept)
    return gradient_intercept


slope_history = []
intercept_history = []
def gradient_decent(x_data,y_data,epochs,learning_rate,initial_slope,initial_intercept):
    slope = initial_slope
    intercept = initial_intercept
    for epoch in range(epochs):
        slope  = slope - learning_rate*(gradient_slope(slope,intercept,x_data,y_data))
        intercept = intercept - learning_rate*(gradient_intercept(slope,intercept,x_data,y_data))
        #print(slope)
        #print(intercept)
        slope_history.append(slope)
        intercept_history.append(intercept)
    return (slope,intercept)

epochs = 1000
learning_rate = 0.0001
initial_slope = 0
initial_intercept = 0

(slope,intercept) = gradient_decent(X_train,y_train,epochs = epochs,learning_rate=learning_rate,initial_slope = initial_slope,initial_intercept =initial_intercept)
m = slope
c = intercept

'''
print(slope)
print(intercept)     
plt.plot(slope_history)
plt.plot(intercept_history)  
#plt.legend()
plt.show()      
'''
 
#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

# Predicting the Results
y_pred = X_test*m + c

# Visualizing the Results
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_pred)
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()

#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#

    
    