# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#import random

# Reading the dataset from data
# 50_Startups dataset
dataset = pd.read_csv(r'D:\Madhus_data\repositories\linear_regression\data\50_Startups.csv')
x_labels = ['R&D Spend', 'Administration', 'Marketing Spend']
y_lables = ['Profit']
x_labels_categorical = ['State']
# Squeezed 3D data for Visualization purpose
#x_labels = random.sample(x_labels,2)
#y_lables = ['Profit']
#x_labels_categorical = []

# Creating Dependent and Independent variables
X = dataset[x_labels]
# Renaming for tensorflow scope
X.columns = ['a', 'b','c']
X = X.values
y = dataset[y_lables].values

# Splitting the data into training set and test set
X_train,X_test = np.split(X,indices_or_sections = [int(len(X)*0.8)])
y_train,y_test = np.split(y,indices_or_sections = [int(len(X)*0.8)])

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

# Variables 
epochs = 100
learning_rate = 0.001

# Feature Columns
feature_columns = [tf.feature_column.numeric_column(key="a"),tf.feature_column.numeric_column(key="b"),tf.feature_column.numeric_column(key="c")]

# Creating feature dictionaries
features_train = {'a':X_train[:,0],'b':X_train[:,1],'c':X_train[:,2]}
features_test  = {'a':X_test[:,0], 'b':X_test[:,1], 'c':X_test[:,2]}

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
y_pred = np.array(y_predicted)

# Visualizing the results
plt.scatter(np.arange(0,len(y_pred),1),y_pred,cmap = 'Sequential')
plt.scatter(np.arange(0,len(y_pred),1),y_test,cmap = 'Sequential')
plt.gca().xaxis.grid(True)
plt.xticks(np.arange(0,len(y_pred),1))
plt.ylabel(y_lables[0])
plt.xlabel('Samples')
plt.show()

'''
if len(x_labels)==2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = X_test[:,0]
    ys = X_test[:,1]
    zs = y_test
    ax.scatter(xs, ys, zs, c='r', marker='o')
    ax.scatter(xs, ys, y_pred, c='b', marker='^')
    ax.set_xlabel(x_labels[0])
    ax.set_ylabel(x_labels[1])
    ax.set_zlabel(y_lables[0])
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    X = np.arange(min(xs), max(xs),(max(xs)-min(xs))/100)
    Y = np.arange(min(ys), max(ys),(max(ys)-min(ys))/100)
    X, Y = np.meshgrid(X, Y)
    Z = lr.predict(np.concatenate((X.ravel().reshape(-1,1),Y.ravel().reshape(-1,1)),axis=1)).reshape(X.shape)
    
    from matplotlib import cm
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel(x_labels[0])
    ax.set_ylabel(x_labels[1])
    ax.set_zlabel(y_lables[0])
    plt.show()
'''


#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#