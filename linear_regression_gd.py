# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:36:58 2023

@author: hoang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def f(x):
    return 2*x + 3

x = np.linspace(0, 20, 50)
y = f(x) + 5*np.random.rand(50)

#x  = x.reshape(-1,1)

#print(x)
#print(y)
"""
mdl = LinearRegression()
mdl.fit(x, y)
print(mdl.coef_)
print(mdl.intercept_)

w0 = mdl.intercept_
w1 = mdl.coef_

plt.scatter(x, y)
yy = x*w1 + w0
plt.plot(x, yy, color='red')
plt.legend(['data', 'fitted line'])
"""

def predict(x, w0, w1):
    return w1*x + w0

def gradient(y_hat, y, x):
    dw1 = 2*x*(y_hat - y)
    dw0 = 2*(y_hat - y)
    return (dw1, dw0)

def update_parameter(w1, w0, lr, dw1, dw0):
    w1_new = w1 - lr*dw1
    w0_new = w0 - lr*dw0
    return (w1_new, w0_new)


w0, w1 = 3, 3


for _ in range(100):
    for i in range(50):
        xx = x[i]
        yy = y[i]
        y_hat = predict(xx, w0, w1)
        dw1, dw0 = gradient(y_hat, yy, xx)
        w1, w0 = update_parameter(w1, w0, 0.001, dw1, dw0)
        
        print('turn', i)
        print(dw1)
        print(dw0)
        print(w1)
        print(w0)
        

plt.scatter(x.reshape(-1,1), y)
yy = x*w1 + w0
plt.plot(x.reshape(-1,1), yy, color='red')
plt.legend(['data', 'fitted line'])




