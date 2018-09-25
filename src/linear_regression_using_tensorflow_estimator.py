# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Reading the dataset from data
dataset = pd.read_csv(r'..\\data\\prices.csv')

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
X_train,X_test = np.split(X,indices_or_sections = [int(len(X)*0.2)])
y_train,y_test = np.split(y,indices_or_sections = [int(len(X)*0.2)])

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

# Variables 
epochs = 100
learning_rate = 0.001

# Feature Columns
feature_columns = [tf.feature_column.numeric_column(key="X")]

# Creating feature dictionaries
features_train = {'X':X_train}
features_test  = {'X':X_test}

# Creating an Input function which would return a batch dataset on every call
def input_function(features, labels, batch_size):
    data = tf.data.Dataset.from_tensor_slices((dict(features), labels))     # Convert the inputs to a Dataset.
    return (data.shuffle(10).batch(5).repeat().make_one_shot_iterator().get_next()) #Returning the batch dataset

# Making the lambda function of train dataset
input_train = lambda: input_function(features_train, y_train,5)

# Build the Estimator.
model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# Train the model.
model.train(input_fn = input_train, steps = epochs)

#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

# Creating a input function for prediction
predict_input_fn = tf.estimator.inputs.numpy_input_fn(features_test, shuffle=False)

# Prediction the results
predict_results = model.predict(input_fn=predict_input_fn) #This yeilds a python generator

# Extracting the y-predicted values into a numpy array
y_predicted = []
for prediction in predict_results:
    y_predicted.append(prediction['predictions'])
y_predicted = np.array(y_predicted)

# Visualizing the Results
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_predicted,c='green')
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()

#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#