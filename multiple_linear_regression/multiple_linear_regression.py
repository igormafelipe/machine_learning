# Multiple Linear Regression
# ----------------------------------
# Cost function: MSE
# Optimization algorithm: Gradient Descent
# ----------------------------------
# Coded by: Igor Mafra Felipe
# 11/25/2023
import numpy as np

def cost_function(x, y, w, b):
    m = x.shape[0]
    
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost += (f_wb_i - y[i]) ** 2

    return cost / (2 * m)

# wrong, not taking number of features into account properly.
def gradient(x, y, w, b):
    m, n = x.shape # m = number of samples, n = number of features
    dj_dw = np.zeros(n)
    dj_db = 0

    # this is wrong
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        for j in range(n):
            dj_dw[j] += (f_wb_i - y[i]) * x[i][j]
        dj_db += f_wb_i - y[i]

    dj_db /= m
    dj_dw /= m
    
    return dj_dw, dj_db
    
# a = learning rate
# num_iter = number of iterations to run for
def multiple_gradient_descent(x, y, w, b=0, a=0.01, num_iter=2000):
    cost_history = []
    for i in range(num_iter):
        dj_dw, dj_db = gradient(x, y, w, b)
        
        w = w - a * dj_dw
        b = b - a * dj_db
        
        cost = cost_function(x, y, w, b)
        cost_history.append(cost)
    
    return w, b, cost_history