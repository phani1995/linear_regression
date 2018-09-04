# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_excel(r'C:\Users\Madhusudhan\linear_regression\data\slr09.xls')

X = dataset['X'].values
y = dataset['Y'].values


X = np.reshape(X,newshape = (-1,1))
y = np.reshape(y,newshape = (-1,1))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X = X, y = y)

y_pred = lr.predict(X)
plt.plot(X,y_pred)