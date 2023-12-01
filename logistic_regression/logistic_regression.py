# Logistic Regression
# ----------------------------------
# Cost function: MSE
# Optimization algorithm: Gradient Descent
# ----------------------------------
# Coded by: Igor Mafra Felipe
# 12/01/2023

import numpy as np

def cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb_i = 1 / (1 + np.exp(-(np.dot(x[i], w) + b)))
        total_cost += y[i] * np.log(f_wb_i) + (1 - y[i]) * np.log(1 - f_wb_i)
    return total_cost / m

def gradient_logistic(x, y, w, b):
    m, n = x.shape
    dj_db = 0
    dj_dw = np.zeros((n,))
    
    for i in range(m):
        f_wb_i = 1 / (1 + np.exp(-(np.dot(x[i], w) + b)))
        dj_db += f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += (f_wb_i - y[i]) * x[i, j]
    
#     return dj_db / m, dj_dw / m

def gradient_descent(x, y, w=0, b=0, a=0.003, num_iter=5000):
    cost_history = []
    
    for i in range(num_iter):
        db, dw = gradient_logistic(x, y, w, b)
        w = w - a * dw
        b = b - a * db
        
        cost_history.append(cost(x, y, w, b))
        
    return w, b, cost_history