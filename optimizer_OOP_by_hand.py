# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:28:39 2023

@author: hoang
"""

import numpy as np
from numpy.random import permutation

class Line(): 
    """
    Linear model with two weights w0, w1: y = w1*x + w0
    w1: slope
    w0: intercept (biases)
    """
    def __init__(self):
        # generate weights 
        self.weights = np.array([np.random.uniform(0,1,1)[0] for _ in range(2)])
        self.derivative_funcs = np.array([self.dx_w0, self.dx_w1])
    
    def evaluate(self, x):
        return self.weights[0] + self.weights[1]*x
    
    
    def derivative(self, x, y):
        partial_derivatives = []
        yhat = self.evaluate(x)
        partial_derivatives.append(self.dx_w0(x, y, yhat))
        partial_derivatives.append(self.dx_w1(x, y, yhat))
        return np.array(partial_derivatives)
    
    def dx_w0(self, x, y, yhat):
        return 2*(yhat - y)
    
    def dx_w1(self, x, y, yhat):
        return 2*x*(yhat - y)
    
    def __str__(self):
        return f"{self.weights}"
        return f"y = {self.weights[0]} + {self.weights[1]}*x" # formatted string literal

def stochastic_sample(xs, ys):
    perm = permutation(len(xs))
    x = xs[perm[0]]
    y = xs[perm[0]]    
    
    return x,y 

def gradient(dx, evaluate, xs, ys):
    N = len(xs)
    total = 0
    for x,y in zip(xs,ys):
        yhat = evaluate(x)
        total = total + dx(x, y, yhat)
    
    gradient = total/N
    return gradient

def gd(model, xs, ys, lr=0.01, max_interator=1000):
    for i in range(max_interator):
        model.weights = [weight - lr*gradient(derivative_func, model.evaluate, xs, ys) for weight, derivative_func in zip(model.weights, model.derivative_funcs)]

def sgd(model, xs, ys, lr=0.01, max_interator=1000):
    for i in range(max_interator):
        x, y = stochastic_sample(xs, ys)
        model.weights = [weight - lr*derivative for weight, derivative in zip(model.weights, model.derivative(x, y))]
    
def momentum(model, xs, ys, decay_factor=0.9, lr=0.01, max_iterator=1000):
    gradients = np.array([0 for _ in range(len(model.weights))])
    #print('gradients', gradients)
    for i in range(max_iterator):
        x, y = stochastic_sample(xs, ys)
        #gradients = [decay_factor*g + lr*derivative for g, derivative in zip(gradients, model.derivative(x, y))]
        #model.weights = [weight - g for weight, g in zip(model.weights, gradients)]
        gradients = decay_factor*gradients + lr*model.derivative(x, y)
        model.weights = model.weights - gradients
        
# create sample
xs = np.linspace(1, 7, 7)
ys = xs
# target intercept = 0, slope = 1

model = Line()
momentum(model, xs, ys)
print(model)
plt.scatter(xs, ys)

w0, w1 = model.weights
plt.plot(xs, w1*xs + w0, color='red')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    