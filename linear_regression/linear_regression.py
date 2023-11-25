# Linear Regression: Predicting House Prices
# ----------------------------------
# Cost function: MSE
# Optimization algorithm: Gradient Descent
# ----------------------------------
# Coded by: Igor Mafra Felipe
# 11/25/2023

import numpy as np

def cost_function(x, y, w, b):
    cost = 0
    m = x.shape[0]
    
    for i in range(m):
        f_x_i = w * x[i] + b
        cost += (f_x_i - y[i]) ** 2
    
    return cost / (2 * m)

def gradient(x, y, w, b):
    dj_dw = 0
    dj_db = 0

    m = x.shape[0]
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]

    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db
    
# a = learning rate
# num_iter = number of iterations to run for
def gradient_descent(x, y, w=0, b=0, a=0.01, num_iter=2000):
    cost_history = []
    for i in range(num_iter):
        dj_dw, dj_db = gradient(x, y, w, b)
        w = w - a * dj_dw
        b = b - a * dj_db
        
        cost = cost_function(x, y, w, b)
        cost_history.append(cost)
    
    return w, b, cost_history