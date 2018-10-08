# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from mpl_toolkits.mplot3d import  Axes3D


# Reading the dataset from data
'''
# Concrete Dataset
dataset = pd.read_csv(r'D:\Madhus_data\repositories\linear_regression\data\Concrete_Data_Yeh.csv')
x_labels = ['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']
y_lables = ['csMPa']
x_labels_categorical = []

# Squeezed 3D data for Visualization purpose
x_labels = random.sample(x_labels,2)
y_lables = ['csMPa']
x_labels_categorical = []

'''

# 50_Startups dataset
dataset = pd.read_csv(r'D:\Madhus_data\repositories\linear_regression\data\50_Startups.csv')
x_labels = ['R&D Spend', 'Administration', 'Marketing Spend']
y_lables = ['Profit']
x_labels_categorical = ['State']

# Squeezed 3D data for Visualization purpose
x_labels = random.sample(x_labels,2)
y_lables = ['Profit']
x_labels_categorical = []


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
lr = LinearRegression(normalize=True)
lr.fit(X = X_train, y = y_train)

#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

# Predicting the Results
y_pred = lr.predict(X_test)

# Visualizing the results
plt.scatter(np.arange(0,len(y_pred[:10]),1),y_pred[:10],cmap = 'Sequential')
plt.scatter(np.arange(0,len(y_test[:10]),1),y_test[:10],cmap = 'Sequential')
plt.gca().xaxis.grid(True)
plt.xticks(np.arange(0,len(y_pred[:10]),1))
plt.ylabel(y_lables[0])
plt.xlabel('Samples')
plt.show()

# 3D visualization 
if len(x_labels)==2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = X_test[:,0]
    ys = X_test[:,1]
    ax.scatter(xs, ys, y_test, c='r', marker='o')
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
    
#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#