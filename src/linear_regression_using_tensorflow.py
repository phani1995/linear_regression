# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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
X_train,X_test = np.split(X,indices_or_sections = [int(len(X)*0.2)])
y_train,y_test = np.split(y,indices_or_sections = [int(len(X)*0.2)])

# Reshaping the numpy arrays since the tensorflow model expects 2-D array in further code
X_train = np.reshape(X_train,newshape = (-1,1)).astype('float32')
y_train = np.reshape(y_train,newshape = (-1,1)).astype('float32')
X_test = np.reshape(X_test,newshape = (-1,1)).astype('float32')
y_test = np.reshape(y_test,newshape = (-1,1)).astype('float32')

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

# Variables for training 
epochs = 1000
learning_rate = 0.0001

# Tensors to build the graph
X_tf = tf.placeholder(tf.float32,shape = (None,1),name = 'x_palceholder')
m = tf.Variable(tf.ones([1,1]))
c = tf.Variable(tf.ones(shape=(1,1),dtype=tf.float32),name='intercept')
y_actual = tf.placeholder(tf.float32,shape = (None,1),name = 'y_actual_palceholder')

# Equation of line in Tensorflow
y_pred = tf.add(tf.matmul(X_tf,m),c)

# Loss Function
loss = tf.reduce_mean(tf.square((y_pred - y_actual)))

# Creating Training step using Gradient Descent Optimizer
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Training 
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epochs):
        sess.run(training_step,feed_dict= {X_tf:X_train,y_actual:y_train})
        
#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

# Predicting the Results (the tensorflow session is still active)        
    y_predicted = sess.run(y_pred,feed_dict= {X_tf:X_test})

# Visualizing the Results
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_predicted,c='green')
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()

#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#
