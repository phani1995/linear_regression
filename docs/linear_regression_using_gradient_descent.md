
# Linear Regression using Gradient Descent from scratch

![Image not found](/assets/images/linear_regression_using_gradient_descent_files/title_image.png)

## The Theory
Linear Regression is the process of fitting a line to the dataset.  

Here we are using gradient decent optimization algorithm. The algorithm is coded from scratch for better understanding. 

![gif not found](/assets/images/linear_regression_using_gradient_descent_files/gradient_descent.gif)
## Single Variable Linear Regression
## The Mathematics
The equation of Line is

$$
y = m*x+c
$$

Where,<br/>
y = dependent variable<br/>
X = independent variable<br/>
C = intercept 

The algorithm is trying to fit a line to the data by adjusting the values of m and c. Its Objective is to attain to a value of m such that for any given value of x it would be properly predicting the value of y.

There are various ways in which we can attain the values of m and c
* Statistical approach
* Iterative approach

Here we are using a scikit learn framework which internally uses iterative approach to attain the linear regression.
## The Dataset 

Dataset consists of two columns namely X and y
Where<br/>

For List Price Vs. Best Price for a New GMC Pickup dataset<br/>
X = List price (in $1000) for a GMC pickup truck<br/>
Y = Best price (in $1000) for a GMC pickup truck<br/>
The data is taken from Consumer’s Digest.

For Fire and Theft in Chicago <br/>
X = fires per 100 housing units<br/>
Y = thefts per 1000 population within the same Zip code in the Chicago metro area<br/>
The data is taken from U.S Commission of Civil Rights.

For Auto Insurance in Sweden dataset<br/>
X = number of claims<br/>
Y = total payment for all the claims in thousands of Swedish Kronor<br/>
The data is taken from Swedish Committee on Analysis of Risk Premium in Motor Insurance.

For Gray Kangaroos dataset<br/>
X = nasal length (mm ¥10)<br/>
Y = nasal width (mm ¥ 10)<br/>
for a male gray kangaroo from a random sample of such animals<br/>
The data is taken from Australian Journal of Zoology, Vol. 28, p607-613.

[Link to All Datasets](http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html)

# The Code

# Imports
Numpy import for array processing, python doesn’t have built in array support. The feature of working with native arrays can be used in python with the help of numpy library.

Pandas is a library of python used for working with tables, on importing the data, mostly data will be of table format, for ease manipulation of tables pandas library is imported

Matplotlib is a library of python used to plot graphs, for the purpose of visualizing the results we would be plotting the results with the help of matplotlib library.

Math import is just used to square the numerical values

FuncAnimation is to create a animation which shows how the line fits with the data.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from matplotlib.animation import FuncAnimation
```

# Reading the dataset from data
In this line of code using the read_excel method of pandas library, the dataset has been imported from data folder and stored in dataset variable.


```python
# Reading the dataset from data
dataset = pd.read_csv(r'..\\data\\auto_insurance.csv')
```

On viewing the dataset, it contains of two columns X and Y where X is dependent variable and Y is Independent Variable.


```python
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108</td>
      <td>392.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>46.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>15.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>124</td>
      <td>422.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>119.4</td>
    </tr>
  </tbody>
</table>
</div>



# Creating Dependent and Independent variables
The X Column from the dataset is extracted into an X variable of type numpy, similarly the y variable
X is an independent variable 
Y is dependent variable Inference


```python
X = dataset['X'].values
y = dataset['Y'].values
```

On execution of first line would result in a pandas Series Object
On using values attribute it would result in an numpy array


```python
print(type(dataset['X']))
print(type(dataset['X'].values))
```

    <class 'pandas.core.series.Series'>
    <class 'numpy.ndarray'>
    

# Visualizing the data 
The step is to just see how the dataset is 
On visualization the data would appear something like this
The X and Y attributes would vary based on dataset.
Each point on the plot is a data point showing the respective Number of Claims on x-axis and Total Payment on y-axis


```python
title='Linear Regression on <Dataset>'
x_axis_label = 'X-value < The corresponding attribute of X in dataset >'
y_axis_label = 'y-value < The corresponding attribute of X in dataset >'
title='Linear Regression on Auto Insurance Sweden Dataset'
x_axis_label = "Total Payment"
y_axis_label = "Number of Claims"

plt.scatter(X,y)
plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()
```


![png](/assets/images/linear_regression_using_gradient_descent_files/linear_regression_using_gradient_descent_11_0.png)


# Splitting the data into training set and test set
We are splitting the whole dataset into training and test set where training set is used for fitting the line to data and test set is used to check how good the line if for the data.


```python
# Splitting the data into training set and test set
X_train,X_test = np.split(X,indices_or_sections = [int(len(X)*0.8)])
y_train,y_test = np.split(y,indices_or_sections = [int(len(X)*0.8)])
```

# Functions for caluculation of gradient descent

# Function to caluculate mean square error
This function takes `slope`,`intercept`,`x_data` and `y_data` and tries to caluculate error between the depent varible and independent varibles based on he mean square error formula

$$mse = \frac{1}{n}\displaystyle\sum_{i=1}^{n}(y_a-y_p)^2$$
where,<br>
$y_a$ is y actual (ground truth)<br>
$y_p$ is y predicted (value predicted with the given slope and intercept values


```python
def mean_square_error(slope,intercept,x_data,y_data):
    n = len(x_data)
    sum_of_squares = 0
    for i in range(n):
        y_pred = x_data[i]*slope+intercept
        y_actual = y_data[i]
        sum_of_squares += (y_actual - y_pred)**2  
        
    mean_square_error = math.sqrt(sum_of_squares/n)
    return mean_square_error
```

# Function to caluculate Gardient of slope at each intermidiatory iteration
This function is to caluculate the first order drivates of the function. This first order derivates are caluculated at every  iteration of a the function.<br>

mse is the loss function and it is sappose to be reduced . In order to reduce it we need to take a gradient step towards the minimum and the direction in which to take step is given by gradients. So we caluculate gradients.<br>
The partial derivative of mse with respective slope is...<br>
$$
 \frac{\partial}{\partial m} =  \frac{1}{n} \displaystyle\sum_{i=1}^{n} -x_i (y_i- (m*x_i+c))
$$
where <br>
m is slope <br>
c is intercept <br>
$x_i$ is independent varible <br>
$y_i$ is dependent varible (ground truth) <br>

In this function we iterate over all the data points to attain the gradient of slope for the given slope and intercept


```python
def gradient_slope(slope,intercept,x_data,y_data):
    n = len(x_data)
    sum_of_gradients = 0
    for i in range(n):
        y_pred = x_data[i]*slope + intercept
        y_actual = y_data[i]
        sum_of_gradients += x_data[i]*(y_actual-y_pred)
    sum_of_gradients = -1*sum_of_gradients
    gradient_slope = sum_of_gradients*(2/n)
    return gradient_slope
```

# Function to caluculate Gardient of intercept at each intermidiatory iteration 

This function is to caluculate the first order drivates of the function. This first order derivates are caluculated at every  iteration of a the function.   mse is the loss function and it is sappose to be reduced.    

In order to reduce it we need to take a gradient step towards the minimum and the direction in which to take step is given by gradients. So we caluculate gradients.<br>
The partial derivative of mse with respective intercept is...<br>
$$
 \frac{\partial}{\partial m} =  \frac{1}{n} \displaystyle\sum_{i=1}^{n} -(y_i- (m*x_i+c))
$$
where <br>
m is slope <br>
c is intercept <br>
$x_i$ is independent varible <br>
$y_i$ is dependent varible (ground truth) <br>

In this function we iterate over all the data points to attain the gradient of intercept for the given slope and intercept


```python
def gradient_intercept(slope,intercept,x_data,y_data):
    n = len(x_data)
    sum_of_gradients = 0
    for i in range(n):
        y_pred = x_data[i]*slope +intercept
        y_actual = y_data[i]
        sum_of_gradients += -1 *(y_actual-y_pred)
    sum_of_gradients = -1*sum_of_gradients
    gradient_intercept = sum_of_gradients*(2/n)
    return gradient_intercept
```

# Gradient Decent function
This function performces gradient descent optimaization for the given epochs withe the given learning rate. Gradient decent is an optimization algoithm used to optimize the given loss function by caluculating the first order derivatives of the varibles in the function and adjusting the varibles with the help of gradients such that the loss function would attain a minimum value.  
Gradient Decent is a first order optimizaion function.   

Args :   
`x_data` :independent varible datapoints   
`y_data` :dependent varible datapoints    
`epochs` :number of times the algorithms need to iterate over the entire data   
`learning_rate` : the magnitude at which the gradient steps to be taken to attain the minimum value of funciton.   
`initial_slope` : staring value of the slope varible   
`initial_intercept`  : starting value of the intercept varible  
Returns: a tuple     
`slope` : final slope value.  
`intercept`: final intercept value.  

Line5,6 : slope and intercept values are initalized with intial values.  
Line7 : iterating over epochs.  
Line 8,9 : updating the varibles with the magnitude of learning rate multiplied with gradient values.   
Line 10,11 : mean square loss is caluculated and printed.  
Line 12,13,14 : varibles are appended into history list for further plotting and visulization.  


```python
slope_history = []
intercept_history = []
loss_history = []
def gradient_decent(x_data,y_data,epochs,learning_rate,initial_slope,initial_intercept):
    slope = initial_slope
    intercept = initial_intercept
    for epoch in range(epochs):
        slope  = slope - learning_rate*(gradient_slope(slope,intercept,x_data,y_data))
        intercept = intercept - learning_rate*(gradient_intercept(slope,intercept,x_data,y_data))
        loss = mean_square_error(slope,intercept,x_data,y_data)
        print("The loss obtained is %0.2f "%(loss))
        loss_history.append(loss)
        slope_history.append(slope)
        intercept_history.append(intercept)
    return (slope,intercept)
```

# Training 
# Hyperparameters
Here we initial all the hyperparameters and run to gradient decent and make changes to hyperparmeters until the funtion reduces its loss.  
First plot visualize the loss value.  
Second plot to visualise the variation in slope and intercept values.  


```python
epochs = 500
learning_rate = 0.00001
initial_slope = 0
initial_intercept = 0

(slope,intercept) = gradient_decent(X_train,y_train,epochs = epochs,learning_rate=learning_rate,initial_slope = initial_slope,initial_intercept =initial_intercept)

# The plot to inspect the decrease in loss
plt.title("Loss Function")
plt.plot(loss_history)
plt.show()


# The plot to inspect the variation in slope and intercept
plt.plot(slope_history)
plt.plot(intercept_history)
plt.show()
     
```

    The loss obtained is 124.91 
    The loss obtained is 122.15 
    The loss obtained is 119.46 
    The loss obtained is 116.84 
    The loss obtained is 114.28 
    The loss obtained is 111.79 
    The loss obtained is 109.36 
    The loss obtained is 107.00 
    The loss obtained is 104.69 
    The loss obtained is 102.44 
    The loss obtained is 100.25 
    The loss obtained is 98.12 
    The loss obtained is 96.04 
    The loss obtained is 94.02 
    The loss obtained is 92.05 
    The loss obtained is 90.13 
    The loss obtained is 88.26 
    The loss obtained is 86.44 
    The loss obtained is 84.67 
    The loss obtained is 82.95 
    The loss obtained is 81.27 
    The loss obtained is 79.64 
    The loss obtained is 78.05 
    The loss obtained is 76.50 
    The loss obtained is 75.00 
    The loss obtained is 73.54 
    The loss obtained is 72.12 
    The loss obtained is 70.74 
    The loss obtained is 69.40 
    The loss obtained is 68.09 
    The loss obtained is 66.82 
    The loss obtained is 65.59 
    The loss obtained is 64.40 
    The loss obtained is 63.23 
    The loss obtained is 62.11 
    The loss obtained is 61.01 
    The loss obtained is 59.95 
    The loss obtained is 58.92 
    The loss obtained is 57.92 
    The loss obtained is 56.95 
    The loss obtained is 56.01 
    The loss obtained is 55.10 
    The loss obtained is 54.22 
    The loss obtained is 53.36 
    The loss obtained is 52.53 
    The loss obtained is 51.73 
    The loss obtained is 50.95 
    The loss obtained is 50.20 
    The loss obtained is 49.47 
    The loss obtained is 48.77 
    The loss obtained is 48.09 
    The loss obtained is 47.43 
    The loss obtained is 46.80 
    The loss obtained is 46.18 
    The loss obtained is 45.59 
    The loss obtained is 45.01 
    The loss obtained is 44.46 
    The loss obtained is 43.93 
    The loss obtained is 43.41 
    The loss obtained is 42.91 
    The loss obtained is 42.43 
    The loss obtained is 41.97 
    The loss obtained is 41.53 
    The loss obtained is 41.10 
    The loss obtained is 40.68 
    The loss obtained is 40.29 
    The loss obtained is 39.90 
    The loss obtained is 39.53 
    The loss obtained is 39.18 
    The loss obtained is 38.84 
    The loss obtained is 38.51 
    The loss obtained is 38.19 
    The loss obtained is 37.89 
    The loss obtained is 37.60 
    The loss obtained is 37.32 
    The loss obtained is 37.05 
    The loss obtained is 36.79 
    The loss obtained is 36.54 
    The loss obtained is 36.30 
    The loss obtained is 36.07 
    The loss obtained is 35.85 
    The loss obtained is 35.64 
    The loss obtained is 35.44 
    The loss obtained is 35.25 
    The loss obtained is 35.06 
    The loss obtained is 34.89 
    The loss obtained is 34.72 
    The loss obtained is 34.55 
    The loss obtained is 34.40 
    The loss obtained is 34.25 
    The loss obtained is 34.10 
    The loss obtained is 33.97 
    The loss obtained is 33.84 
    The loss obtained is 33.71 
    The loss obtained is 33.59 
    The loss obtained is 33.47 
    The loss obtained is 33.36 
    The loss obtained is 33.26 
    The loss obtained is 33.16 
    The loss obtained is 33.06 
    The loss obtained is 32.97 
    The loss obtained is 32.88 
    The loss obtained is 32.80 
    The loss obtained is 32.72 
    The loss obtained is 32.64 
    The loss obtained is 32.57 
    The loss obtained is 32.50 
    The loss obtained is 32.43 
    The loss obtained is 32.37 
    The loss obtained is 32.31 
    The loss obtained is 32.25 
    The loss obtained is 32.19 
    The loss obtained is 32.14 
    The loss obtained is 32.09 
    The loss obtained is 32.04 
    The loss obtained is 31.99 
    The loss obtained is 31.95 
    The loss obtained is 31.91 
    The loss obtained is 31.86 
    The loss obtained is 31.83 
    The loss obtained is 31.79 
    The loss obtained is 31.75 
    The loss obtained is 31.72 
    The loss obtained is 31.69 
    The loss obtained is 31.66 
    The loss obtained is 31.63 
    The loss obtained is 31.60 
    The loss obtained is 31.57 
    The loss obtained is 31.55 
    The loss obtained is 31.53 
    The loss obtained is 31.50 
    The loss obtained is 31.48 
    The loss obtained is 31.46 
    The loss obtained is 31.44 
    The loss obtained is 31.42 
    The loss obtained is 31.40 
    The loss obtained is 31.38 
    The loss obtained is 31.37 
    The loss obtained is 31.35 
    The loss obtained is 31.34 
    The loss obtained is 31.32 
    The loss obtained is 31.31 
    The loss obtained is 31.30 
    The loss obtained is 31.28 
    The loss obtained is 31.27 
    The loss obtained is 31.26 
    The loss obtained is 31.25 
    The loss obtained is 31.24 
    The loss obtained is 31.23 
    The loss obtained is 31.22 
    The loss obtained is 31.21 
    The loss obtained is 31.20 
    The loss obtained is 31.19 
    The loss obtained is 31.19 
    The loss obtained is 31.18 
    The loss obtained is 31.17 
    The loss obtained is 31.17 
    The loss obtained is 31.16 
    The loss obtained is 31.15 
    The loss obtained is 31.15 
    The loss obtained is 31.14 
    The loss obtained is 31.14 
    The loss obtained is 31.13 
    The loss obtained is 31.13 
    The loss obtained is 31.12 
    The loss obtained is 31.12 
    The loss obtained is 31.11 
    The loss obtained is 31.11 
    The loss obtained is 31.11 
    The loss obtained is 31.10 
    The loss obtained is 31.10 
    The loss obtained is 31.10 
    The loss obtained is 31.09 
    The loss obtained is 31.09 
    The loss obtained is 31.09 
    The loss obtained is 31.08 
    The loss obtained is 31.08 
    The loss obtained is 31.08 
    The loss obtained is 31.08 
    The loss obtained is 31.07 
    The loss obtained is 31.07 
    The loss obtained is 31.07 
    The loss obtained is 31.07 
    The loss obtained is 31.07 
    The loss obtained is 31.06 
    The loss obtained is 31.06 
    The loss obtained is 31.06 
    The loss obtained is 31.06 
    The loss obtained is 31.06 
    The loss obtained is 31.06 
    The loss obtained is 31.06 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.05 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.03 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    The loss obtained is 31.04 
    


![png](/assets/images/linear_regression_using_gradient_descent_files/linear_regression_using_gradient_descent_23_1.png)



![png](/assets/images/linear_regression_using_gradient_descent_files/linear_regression_using_gradient_descent_23_2.png)


# Predicting the Results
based on the obtained slope and intercept values we are precting the values for the test set data.


```python
m = slope
c = intercept

# Predicting the Results
y_pred = X_test*m + c
```

# Visualizing the Results
As we have predicted the y-values for a set of x-values we are visualizing the results to check how good did our line fit for our predictions.
The plot shows the red points are the data points are actual values where the blue line is the predictions.


```python
# Visualizing the Results
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_pred)

plt.title(title)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.show()
```


![png](/assets/images/linear_regression_using_gradient_descent_files/linear_regression_using_gradient_descent_27_0.png)


# Animation of Gradient Descent
This animation to show how on each iteration the line is trying to fit the data. 
`init`: this function to initialize all the values.  
`update`: this funciton to update all the values of plot for each frame.  
> Note: on running the code snippet `.gif` file would be created in the current working directory


```python

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'b', animated=True)
sc = plt.scatter(X_test,y_test,c='red',animated=True)

def init():
    ax.set_title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    ax.set_xlim(min(X_test)-10, max(X_test)+10)
    ax.set_ylim(min(y_test)-10, max(y_test)+10)
    return ln,

def update(frame):
    i = frame 
    m = slope_history[i]
    c = intercept_history[i]   
    y_pred = X_test*m + c
    loss = mean_square_error(m,c,X_test,y_pred)
    ax.text(2,3,loss)
    ln.set_data(X_test, y_pred)
    
anim = FuncAnimation(fig, update, frames=range(len(slope_history)),init_func=init)
anim.save('gradient_descent.gif', fps=30)
plt.show()
print('gif is created')
```

    MovieWriter ffmpeg unavailable.
    
