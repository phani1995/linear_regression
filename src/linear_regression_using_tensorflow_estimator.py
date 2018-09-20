# -*- coding: utf-8 -*-

#-------------------------------------- DATA PREPROCESSING ---------------------------------#

#Imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#Reading the dataset from data
dataset = pd.read_csv(r'..\\data\\prices.csv')

#Creating Dependent and Independent variables
X = dataset['X'].values
y = dataset['Y'].values

#Visualizing the data 
title='Linear Regression on <Dataset>'
x_axis_label = 'X-value < The corresponding attribute of X in dataset >'
y_axis_label = 'y-value < The corresponding attribute of X in dataset >'
plt.scatter(X,y)
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()


#Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y, test_size = 0.8)

'''
# Reshaping the numpy arrays since the tensorflow model expects 2-D array in further code
X_train = np.reshape(X_train,newshape = (-1,1)).astype('float32')
y_train = np.reshape(y_train,newshape = (-1,1)).astype('float32')
X_test = np.reshape(X_test,newshape = (-1,1)).astype('float32')
y_test = np.reshape(y_test,newshape = (-1,1)).astype('float32')
'''

#------------------------------------ DATA PREPROCESSING ENDS -----------------------------#

#--------------------------------------- TRAINING   ---------------------------------------#

#Variables 
epochs = 1000
learning_rate = 0.0001

feature_columns = [tf.feature_column.numeric_column(key="X")]

features = {'X':X_train}
features_test = {'X':X_test}

def input_function(features, labels, batch_size):
   
    # Convert the inputs to a Dataset.
    data = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    '''
    #data = data.map(input_parser)
    data = data.batch(1)
    iterator = data.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return {'X': features}, labels
    '''
    return (data.shuffle(10).batch(5).repeat().make_one_shot_iterator().get_next())

input_train = lambda: input_function(features, y_train,5)
input_test  = lambda: input_function(features_test, y_test,5)

def input_test():
    features = {'X': X_test}
    labels = y_test
    return features, labels

# Build the Estimator.
model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# Train the model.
model.train(input_fn = input_train, steps = epochs)
print("----------------------------")
print(input_train)
print('Training Done')

'''
# Evaluate how the model performs on data it has not yet seen.
eval_result = model.evaluate(input_fn = input_test)
print('Evaluation Done')
'''  
#-------------------------------------- TRAINING ENDS  ------------------------------------#

#------------------------------- PREDICTION AND PLOTING -----------------------------------#

input_dict = {'X':X_test}
predict_input_fn = tf.estimator.inputs.numpy_input_fn(input_dict, shuffle=False)
predict_results = model.predict(input_fn=predict_input_fn)
print(predict_results)
y_predicted =[]
for i, prediction in enumerate(predict_results):
    y_predicted.append(prediction['predictions'])
    

#Visualizing the Results
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_predicted,c='green')
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()

#------------------------------ PREDICTION AND PLOTING ENDS--------------------------------#